import gc
import os
import json
import utils
import torch
import wandb
from tqdm.auto import tqdm
from datasets import load_dataset
from transformers import TextStreamer
from unsloth import FastLanguageModel
from trl import ORPOConfig, ORPOTrainer
from unsloth import is_bfloat16_supported


def ORPO_train(args, output_dir):
    wandb.login(key=args.wandb_token)
    if args.test_name == "":
        name = f"{args.exp_name}_{args.model_name.split('/')[1]}"
    else:
        name = f"{args.exp_name}_{args.model_name.split('/')[1]}_{args.test_name}"
    wandb.init(project="hw6_rlhf", name=name)

    torch_dtype = torch.bfloat16 if is_bfloat16_supported() else torch.float16
    # max_seq_length = 2048
    max_seq_length = args.max_length

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device\n")

    # Model
    model, tokenizer = FastLanguageModel.from_pretrained(
        args.model_name, dtype=torch_dtype, max_seq_length=max_seq_length, load_in_4bit=True
    )
    # utils.YOUR_CODE_HERE

    # Load dataset
    # ================================DO NOT CHANGE!================================
    dataset = load_dataset("Intel/orca_dpo_pairs", split="train")
    EOS_TOKEN = tokenizer.eos_token  # Must add EOS_TOKEN
    dataset = dataset.map(utils.format_prompt, fn_kwargs={"EOS_TOKEN": EOS_TOKEN})
    dataset = dataset.train_test_split(test_size=0.01)

    with open("./test_prompt.json", "r") as f:
        test_data = json.load(f)
    # ================================DO NOT CHANGE!================================

    # Perform model patching and add fast LoRA weights
    # model = FastLanguageModel.get_peft_model(model,...)
    model = FastLanguageModel.get_peft_model(
        model,
        r=64,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=64,
        lora_dropout=0,  # Supports any, but = 0 is optimized
        bias="none",  # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
        random_state=args.seed,
        max_seq_length=max_seq_length,
    )
    # utils.YOUR_CODE_HERE

    # Training arguments
    training_args = ORPOConfig(
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        beta=args.beta,
        learning_rate=args.lr,
        lr_scheduler_type=args.lr_scheduler_type,
        max_steps=args.max_steps,
        num_train_epochs=args.num_epochs,
        optim=args.optimizer,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        warmup_ratio=args.warmup_ratio,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_prompt_length,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_strategy=args.logging_strategy,
        logging_steps=args.logging_steps,
        evaluation_strategy=args.evaluation_strategy,
        eval_steps=args.eval_steps,
        output_dir=output_dir,
        save_strategy=args.save_strategy,
        # save_steps=50,
        report_to=args.report_to,
    )

    # Trainer
    orpo_trainer = ORPOTrainer(
        model=model,
        tokenizer=tokenizer,
        # train_dataset=utils.YOUR_CODE_HERE,
        # eval_dataset=utils.YOUR_CODE_HERE,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        args=training_args,
    )

    # Fine-tune model with ORPO
    orpo_trainer.train()

    # Save model
    model.save_pretrained(output_dir)

    # Inference
    FastLanguageModel.for_inference(model)
    text_streamer = TextStreamer(tokenizer)

    output_data = []

    for data in tqdm(test_data):
        print("=============Generated Answer After Fine-tuning=============\n")
        print(f"Question {data['id']}:\n" + data["prompt"])
        prompt = utils.alpaca_prompt.format(
            "You are a helpful assistant chatbot.",  # Instruction
            data["prompt"],  # Input
            "",  # Response, leave empty for generation
        )
        prompt = tokenizer(prompt, return_tensors="pt").to("cuda")
        generated_sequences = model.generate(**prompt, streamer=text_streamer, max_new_tokens=500)
        # Decode the generated output
        generated_text = tokenizer.batch_decode(generated_sequences, skip_special_tokens=True)[0]
        print("==============================================================\n")

        # Store the output in a list
        output_data.append({"id": data["id"], "prompt": data["prompt"], "generated_text": generated_text})

    # Ensure the submission directory exists
    submission_dir = "submission"
    if not os.path.exists(submission_dir):
        os.makedirs(submission_dir)

    # Write the output data to a JSON file
    # output_file = os.path.join(submission_dir, f"ORPO_{args.model_name.split('/')[1]}.json")
    output_file = os.path.join(submission_dir, f"{name}.json")
    utils.write_json(output_data, output_file)
    # Flush memory
    del orpo_trainer, model
    gc.collect()
    torch.cuda.empty_cache()
