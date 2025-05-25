# train_and_generate.py

import os
import torch
from datasets import Dataset
from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    GPT2Tokenizer,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    pipeline,
    AutoTokenizer
)

# —— 环境配置 ——
# 关闭 HF Hub 符号链接警告（Windows）
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# —— 学号后两位拆分函数 ——
def select_sentences(last_two: int):
    movie_reviews = {
        0: "这部电影太精彩了，节奏紧凑毫无冷场，完全沉浸其中！",
        1: "剧情设定新颖不落俗套，每个转折都让人惊喜。",
        2: "导演功力深厚，镜头语言非常有张力，每一帧都值得回味。",
        3: "美术、服装、布景细节丰富，完全是视觉盛宴！",
        4: "是近年来最值得一看的国产佳作，强烈推荐！",
        5: "剧情拖沓冗长，中途几次差点睡着。",
        6: "演员表演浮夸，完全无法让人产生代入感。",
        7: "剧情老套，充满套路和硬凹的感动。",
        8: "对白尴尬，像是AI自动生成的剧本。",
        9: "看完只觉得浪费了两个小时，再也不想看第二遍。"
    }
    food_reviews = {
        0: "食物完全凉了，吃起来像隔夜饭，体验极差。",
        1: "汤汁洒得到处都是，包装太随便了。",
        2: "味道非常一般，跟评论区说的完全不一样。",
        3: "分量太少了，照片看着满满的，实际就几口。",
        4: "食材不新鲜，有异味，感觉不太卫生。",
        5: "食物份量十足，性价比超高，吃得很满足！",
        6: "味道超级赞，和店里堂食一样好吃，五星好评！",
        7: "这家店口味稳定，已经回购好几次了，值得信赖！",
        8: "点单备注有按要求做，服务意识很棒。",
        9: "包装环保、整洁美观，整体体验非常好。"
    }
    first = last_two % 10
    second = (last_two // 10) % 10
    return movie_reviews[first], food_reviews[second]

if __name__ == "__main__":
    # —— 1. 选句与标签准备 ——
    stu_id = 45
    last_two = stu_id % 100
    movie_text, food_text = select_sentences(last_two)
    texts = [movie_text, food_text]
    labels = [1 if idx < 5 else 0 for idx in [last_two % 10, last_two // 10]]

    # —— 2. 构建数据集 ——
    ds = Dataset.from_dict({"text": texts, "label": labels})

    # —— 3. BERT 微调 ——
    tokenizer_bert = BertTokenizerFast.from_pretrained("bert-base-chinese")
    model_bert = BertForSequenceClassification.from_pretrained(
        "bert-base-chinese", num_labels=2
    )
    def tok_fn(x): return tokenizer_bert(x["text"], truncation=True, padding=False)
    tokenized = ds.map(tok_fn, batched=True)
    collator = DataCollatorWithPadding(tokenizer_bert)
    args = TrainingArguments(
        output_dir="./bert_outputs", num_train_epochs=3,
        per_device_train_batch_size=4, logging_steps=10, save_total_limit=1
    )
    trainer = Trainer(
        model=model_bert, args=args,
        train_dataset=tokenized,
        data_collator=collator,
        tokenizer=tokenizer_bert
    )
    trainer.train()
    preds = trainer.predict(tokenized)
    pred_labels = torch.argmax(torch.tensor(preds.predictions), dim=1)
    print("\n--- 情感分类结果 ---")
    for t, p in zip(texts, pred_labels):
        print(f"{t}\n=> {'正面' if p==1 else '负面'}\n")

    # —— 4. GPT2 文本续写 ——
    # 方法 A：慢版 GPT2Tokenizer
    # tokenizer_gpt = GPT2Tokenizer.from_pretrained(
    #     "uer/gpt2-chinese-cluecorpussmall", use_fast=False
    # )
    # 方法 B（可选）：AutoTokenizer 自动选择
    tokenizer_gpt = AutoTokenizer.from_pretrained(
        "uer/gpt2-chinese-cluecorpussmall", use_fast=False
    )
    model_gpt = GPT2LMHeadModel.from_pretrained("uer/gpt2-chinese-cluecorpussmall")
    gen_pipe = pipeline(
        "text-generation", model=model_gpt,
        tokenizer=tokenizer_gpt, max_length=80, num_return_sequences=1
    )
    prompts = [
        "如果我拥有一台时间机器", "当人类第一次踏上火星", "如果动物会说话，它们最想告诉人类的是",
        "有一天，城市突然停电了", "当我醒来，发现自己变成了一本书", "假如我能隐身一天，我会",
        "我走进了那扇从未打开过的门", "在一个没有网络的世界里", "如果世界上只剩下我一个人",
        "梦中醒来，一切都变了模样"
    ]
    starter = prompts[last_two % 10]
    print("--- 文本续写提示：", starter)
    completion = gen_pipe(starter)[0]["generated_text"]
    print("\n--- 续写结果 ---\n", completion)
