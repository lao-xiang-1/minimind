import time
import argparse
import random
import warnings
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, get_linear_schedule_with_warmup
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from model.model_lora import *
from trainer.trainer_utils import setup_seed, get_model_params
warnings.filterwarnings('ignore')

def init_model(args, training=False):
    """初始化模型，支持训练和推理模式"""
    tokenizer = AutoTokenizer.from_pretrained(args.load_from)
    
    if 'model' in args.load_from:
        model = MiniMindForCausalLM(MiniMindConfig(
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            use_moe=bool(args.use_moe),
            inference_rope_scaling=args.inference_rope_scaling
        ))
        moe_suffix = '_moe' if args.use_moe else ''
        if int(args.start_step) > 0:
            ckp = f'./{args.save_dir}/checkpoint_step_{args.start_step}{moe_suffix}.pth'
        else:
            ckp = f'./{args.save_dir}/full_sft_512{moe_suffix}.pth'
        model.load_state_dict(torch.load(ckp, map_location=args.device), strict=True)
        if args.lora_weight != 'None':
            apply_lora(model)
            load_lora(model, f'./{args.save_dir}/lora/{args.lora_weight}_{args.hidden_size}.pth')
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.load_from, 
            trust_remote_code=True,
            torch_dtype=torch.float16 if args.fp16 else torch.float32
        )
    
    get_model_params(model, model.config)
    
    if training:
        model.train()
        # 如果是LoRA训练，只训练LoRA参数
        if args.lora_weight != 'None' and args.train_lora:
            for name, param in model.named_parameters():
                if 'lora' not in name.lower():
                    param.requires_grad = False
    else:
        model.eval()
        
    return model.to(args.device), tokenizer

def prepare_training_data(tokenizer, conversation, args):
    """准备训练数据，基于对话历史"""
    # 构建完整的对话文本
    full_text = tokenizer.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=False
    )
    
    # 对文本进行分词
    inputs = tokenizer(
        full_text, 
        return_tensors="pt", 
        truncation=True, 
        max_length=2048
    ).to(args.device)
    
    # 创建标签
    labels = inputs["input_ids"].clone()
    
    return {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "labels": labels
    }

def train_on_conversation(model, conversation, tokenizer, args, optimizer, scheduler, scaler=None):
    """在单轮对话上进行训练"""
    if not conversation or len(conversation) < 2:
        return 0.0
    
    # 准备训练数据
    batch = prepare_training_data(tokenizer, conversation, args)
    
    # 训练步骤
    model.train()
    
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]
    
    if scaler is not None and args.device == 'cuda':
        with torch.cuda.amp.autocast():
            # 前向传播
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
    else:
        # 前向传播
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
    
    loss = outputs.loss
    
    if loss is not None:
        # 反向传播
        if scaler is not None and args.device == 'cuda':
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # 梯度裁剪
        if args.max_grad_norm > 0:
            if scaler is not None and args.device == 'cuda':
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        
        # 参数更新
        if scaler is not None and args.device == 'cuda':
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        
        if scheduler is not None:
            scheduler.step()
        
        optimizer.zero_grad()
        
        return loss.item()
    
    return 0.0

def generate_response(model, tokenizer, conversation, args, streamer=None, training_mode=False):
    """生成回复"""
    # 准备输入
    current_conversation = conversation[-args.historys*2:] if args.historys > 0 else conversation
    
    templates = {
        "conversation": current_conversation,
        "tokenize": False, 
        "add_generation_prompt": True
    }
    
    if args.weight == 'reason': 
        templates["enable_thinking"] = True
    
    inputs_text = tokenizer.apply_chat_template(**templates) if args.weight != 'pretrain' else (tokenizer.bos_token + current_conversation[-1]["content"])
    inputs = tokenizer(inputs_text, return_tensors="pt", truncation=True).to(args.device)
    
    # 生成参数
    generation_kwargs = {
        "inputs": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "max_new_tokens": args.max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "repetition_penalty": args.repetition_penalty,
    }
    
    # 在训练模式下，我们需要确保生成时不计算梯度
    if training_mode:
        model.eval()
    
    # 生成回复
    if streamer is not None:
        generation_kwargs["streamer"] = streamer
        with torch.no_grad():
            generated_ids = model.generate(
                **generation_kwargs,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p
            )
    else:
        with torch.no_grad():
            generated_ids = model.generate(
                **generation_kwargs,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p
            )
    
    response = tokenizer.decode(generated_ids[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
    
    # 如果是训练模式，切换回训练状态
    if training_mode:
        model.train()
    
    return response, len(generated_ids[0]) - len(inputs["input_ids"][0])

def main():
    parser = argparse.ArgumentParser(description="MiniMind模型推理与在线训练")
    
    # 模型参数
    parser.add_argument('--load_from', default='model', type=str, help="模型加载路径")
    parser.add_argument('--save_dir', default='out', type=str, help="模型权重目录")
    parser.add_argument('--start_step', default='0', type=str, help="权重文件名")
    parser.add_argument('--weight', default='full_sft', type=str, help="权重名称前缀")
    parser.add_argument('--lora_weight', default='None', type=str, help="LoRA权重名称")
    parser.add_argument('--hidden_size', default=512, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构")
    parser.add_argument('--inference_rope_scaling', default=False, action='store_true', help="启用RoPE位置编码外推")
    
    # 生成参数
    parser.add_argument('--max_new_tokens', default=8192, type=int, help="最大生成长度")
    parser.add_argument('--temperature', default=0.85, type=float, help="生成温度")
    parser.add_argument('--top_p', default=0.85, type=float, help="nucleus采样阈值")
    parser.add_argument('--repetition_penalty', default=1.0, type=float, help="重复惩罚系数")
    parser.add_argument('--historys', default=0, type=int, help="携带历史对话轮数")
    
    # 训练参数
    parser.add_argument('--train', action='store_true', help="启用在线训练模式")
    parser.add_argument('--train_lora', action='store_true', help="仅训练LoRA参数")
    parser.add_argument('--learning_rate', default=1e-5, type=float, help="学习率")
    parser.add_argument('--batch_size', default=1, type=int, help="批大小（在线训练通常为1）")
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int, help="梯度累积步数")
    parser.add_argument('--max_grad_norm', default=1.0, type=float, help="梯度裁剪阈值")
    parser.add_argument('--num_train_epochs', default=1, type=int, help="训练轮数（对历史对话）")
    parser.add_argument('--warmup_steps', default=10, type=int, help="预热步数")
    parser.add_argument('--fp16', action='store_true', help="使用混合精度训练")
    parser.add_argument('--save_steps', default=5, type=int, help="保存步数间隔")
    parser.add_argument('--train_after_generate', action='store_true', default=True, help="生成后立即训练")
    parser.add_argument('--train_strategy', default='online', type=str, choices=['online', 'batch', 'memory'], 
                       help="训练策略: online=在线学习, batch=批次训练, memory=记忆回放")
    parser.add_argument('--memory_size', default=100, type=int, help="记忆回放缓冲区大小")
    parser.add_argument('--human_feedback', action='store_true', help="启用人工反馈训练")
    
    # 其他参数
    parser.add_argument('--show_speed', default=1, type=int, help="显示decode速度")
    parser.add_argument('--show_loss', action='store_true', help="显示训练损失")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help="运行设备")
    
    args = parser.parse_args()
    
    # 默认提示词
    prompts = [
        '你有什么特长？',
        '为什么天空是蓝色的',
        '请用Python写一个计算斐波那契数列的函数',
        '解释一下"光合作用"的基本过程',
        '如果明天下雨，我应该如何出门',
        '比较一下猫和狗作为宠物的优缺点',
        '解释什么是机器学习',
        '推荐一些中国的美食'
    ]
    
    # 初始化模型
    model, tokenizer = init_model(args, training=args.train)
    
    # 初始化优化器和调度器
    optimizer, scheduler, scaler = None, None, None
    conversation_history = []  # 存储所有对话历史
    memory_buffer = []  # 记忆回放缓冲区
    
    if args.train:
        # 准备优化器
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = AdamW(trainable_params, lr=args.learning_rate)
        
        # 学习率调度器
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=1000  # 初始估计，实际会动态调整
        )
        
        # 混合精度训练
        if args.fp16 and args.device == 'cuda':
            scaler = torch.cuda.amp.GradScaler()
        
        print("✅ 训练模式已启用")
        if args.train_lora:
            print("🔧 仅训练LoRA参数")
        if args.human_feedback:
            print("👤 人工反馈训练已启用")
        print(f"📊 学习率: {args.learning_rate}")
        print(f"💾 模型将每 {args.save_steps} 步保存一次")
    
    # 选择输入模式
    print("\n" + "="*50)
    print("MiniMind 交互式对话系统")
    print("="*50)
    
    # 始终初始化streamer，确保输出显示
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    mode = int(input('[0] 自动测试\n[1] 手动对话\n选择模式: '))
    
    # 从已加载 checkpoint 的 step 开始累计
    step = int(getattr(args, 'start_step', 0))
    
    # 对话循环
    while True:
        if mode == 0:
            # 自动测试模式
            if step >= len(prompts):
                print("\n⚠️ 所有测试提示词已完成")
                cont = input("继续手动对话? (y/n): ")
                if cont.lower() == 'y':
                    mode = 1
                else:
                    break
            
            prompt = prompts[step]
            print(f'\n💬 用户: {prompt}')
            use_model_response = True
            
        else:
            # 手动对话模式
            user_input = input('\n💬 用户: ')
            
            # 处理特殊命令
            if user_input.lower() in ['退出', 'exit', 'quit', 'q']:
                break
            elif user_input.lower() in ['保存', 'save']:
                if args.train:
                    save_path = f"./{args.save_dir}/manual_save_step_{step}.pth"
                    torch.save(model.state_dict(), save_path)
                    print(f"✅ 模型已手动保存到: {save_path}")
                continue
            elif user_input.lower() in ['状态', 'status']:
                print(f"\n📊 当前状态:")
                print(f"   对话轮数: {len(conversation_history)//2}")
                print(f"   训练步数: {step}")
                print(f"   内存缓冲区大小: {len(memory_buffer)}")
                if optimizer is not None:
                    print(f"   学习率: {optimizer.param_groups[0]['lr']:.2e}")
                continue
            elif user_input.lower().startswith('系统:'):
                # 系统指令
                if '清空历史' in user_input:
                    conversation_history = []
                    print("🗑️ 对话历史已清空")
                elif '重置模型' in user_input:
                    model, tokenizer = init_model(args, training=args.train)
                    print("🔄 模型已重置")
                continue
            
            # 检查是否包含期望回答（格式：问题||期望回答）
            if '||' in user_input and args.train and args.human_feedback:
                parts = user_input.split('||', 1)
                prompt = parts[0].strip()
                expected_response = parts[1].strip()
                use_model_response = False
                print(f'\n💬 用户: {prompt}')
                print(f'🎯 期望回答: {expected_response}')
            else:
                prompt = user_input
                use_model_response = True
        
        setup_seed(2026 + step)  # 设置随机种子
        
        # 添加用户输入到对话历史
        conversation_history.append({"role": "user", "content": prompt})
        
        # 生成回复
        if use_model_response or not args.train:
            # 使用模型生成回复
            print('🤖 助手: ', end='', flush=True)
            st = time.time()
            
            response, gen_tokens = generate_response(
                model, tokenizer, 
                conversation_history, 
                args, 
                streamer,
                training_mode=args.train
            )
            
            et = time.time()
            
            # 如果是训练模式且有用户反馈选项，询问用户是否满意
            if args.train and args.human_feedback and mode == 1 and use_model_response:
                print("\n" + "-"*40)
                feedback = input("您对模型的回复满意吗？(y/n/输入正确回答): ").strip()
                
                if feedback.lower() == 'n':
                    # 用户不满意，让用户输入正确回答
                    correct_response = input("请输入正确的回答: ").strip()
                    if correct_response:
                        response = correct_response
                        print(f"✅ 已更新回答为: {response}")
                elif feedback.lower() != 'y' and feedback:
                    # 用户直接输入了正确回答
                    response = feedback
                    print(f"✅ 已使用您的回答: {response}")
        else:
            # 使用用户提供的期望回答
            response = expected_response
            print(f'🤖 助手: {response}')
            et = time.time()
            st = et - 1  # 避免除零
            gen_tokens = len(response.split())  # 估算token数
        
        # 添加助手回复到对话历史
        conversation_history.append({"role": "assistant", "content": response})
        
        # 显示生成速度
        if args.show_speed and use_model_response:
            speed = gen_tokens / (et - st) if (et - st) > 0 else 0
            print(f'\n⚡ 速度: {speed:.2f} tokens/s')
        
        # 训练（如果启用）
        if args.train and args.train_after_generate:
            loss = 0.0
            
            if args.train_strategy == 'online':
                # 在线学习：使用当前对话进行训练
                current_conversation = conversation_history[-2:]  # 只使用最近一轮对话
                loss = train_on_conversation(
                    model, current_conversation, tokenizer, args, 
                    optimizer, scheduler, scaler
                )
                
            elif args.train_strategy == 'batch':
                # 批次训练：使用所有历史对话
                if len(conversation_history) >= 4:  # 至少两轮对话
                    loss = train_on_conversation(
                        model, conversation_history, tokenizer, args, 
                        optimizer, scheduler, scaler
                    )
                    
            elif args.train_strategy == 'memory':
                # 记忆回放：保存到缓冲区并随机采样
                memory_buffer.append(conversation_history[-2:])  # 保存当前对话
                if len(memory_buffer) > args.memory_size:
                    memory_buffer.pop(0)  # 保持缓冲区大小
                
                # 从缓冲区随机采样
                if len(memory_buffer) >= args.batch_size:
                    batch = random.sample(memory_buffer, min(args.batch_size, len(memory_buffer)))
                    # 这里简化处理，实际应该合并多个对话为一个batch
                    for conv in batch:
                        loss += train_on_conversation(
                            model, conv, tokenizer, args, 
                            optimizer, scheduler, scaler
                        )
                    loss /= len(batch)
            
            step += 1
            
            # 显示损失
            if args.show_loss and loss > 0:
                print(f'📉 损失: {loss:.4f}')
            
            # 保存检查点
            if step % args.save_steps == 0 and step > 0:
                checkpoint_path = f"./{args.save_dir}/checkpoint_step_{step}.pth"
                if args.lora_weight != 'None' and args.train_lora:
                    # 只保存LoRA权重
                    lora_state_dict = {k: v for k, v in model.state_dict().items() if 'lora' in k.lower()}
                    torch.save(lora_state_dict, checkpoint_path.replace('.pth', '_lora.pth'))
                    print(f"💾 LoRA权重已保存到: {checkpoint_path.replace('.pth', '_lora.pth')}")
                else:
                    torch.save(model.state_dict(), checkpoint_path)
                    print(f"💾 模型已保存到: {checkpoint_path}")
        
        # 限制对话历史长度
        if args.historys > 0 and len(conversation_history) > args.historys * 2:
            conversation_history = conversation_history[-args.historys * 2:]
    
    # 最终保存
    if args.train and step > 0:
        final_path = f"./{args.save_dir}/checkpoint_step_{step}.pth"
        if args.lora_weight != 'None' and args.train_lora:
            lora_state_dict = {k: v for k, v in model.state_dict().items() if 'lora' in k.lower()}
            torch.save(lora_state_dict, final_path.replace('.pth', '_lora.pth'))
            print(f"\n✅ 最终LoRA权重已保存到: {final_path.replace('.pth', '_lora.pth')}")
        else:
            torch.save(model.state_dict(), final_path)
            print(f"\n✅ 最终模型已保存到: {final_path}")
        
        # 保存对话历史
        if conversation_history:
            import json
            history_path = f"./{args.save_dir}/conversation_history.json"
            with open(history_path, 'w', encoding='utf-8') as f:
                json.dump(conversation_history, f, ensure_ascii=False, indent=2)
            print(f"💬 对话历史已保存到: {history_path}")

if __name__ == "__main__":
    main()