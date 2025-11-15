# ai_agent_human-ojbect-interaction
我现在要写一个ai agent human-object interaction的程序，达到的效果就是使用ai agent去自动输出图像中存在的所有人-物交互对，每一对人-物交互对包括了人的位置、物体的位置、物体的类别以及交互的类别，图像中可能仅存在一个人-物交互对，也可能存在多个。
现在我已经写出了主要的代码，见hio_multi_agent_systemv6.py和hoi_complete_module.py文件，你需要仔细阅读一下这两个文件，以hio_multi_agent_systemv6.py为主。
之后，你需要帮我改进hio_multi_agent_systemv6.py几个地方，如下：
1. 关于使用llm部分，目前使用的是microsoft/phi-2，但是这个模型很不好用，使用的prompt如下：
 prompt = f"""You are a HOI (Human-Object Interaction) expert. Analyze the interaction.

Action: {verb}
Object: {object_class}

Please respond in JSON format with these fields:
- plausibility: score from 0 to 1
- physical_possible: true or false
- common_sense: true or false
- safe: true or false
- reasoning: brief explanation

Example:
{{"plausibility": 0.9, "physical_possible": true, "common_sense": true, "safe": true, "reasoning": "Common interaction"}}

Now analyze: Can a person {verb} a {object_class}?
Response:"""
但是我发现使用prompt提示后根本输出不了我想要的格式，如{{"plausibility": 0.9, "physical_possible": true, "common_sense": true, "safe": true, "reasoning": "Common interaction"}}，他每次都乱回答，所以，你帮我换个其他的好用的问答模型，能指定输出格式为json的。
2. 这个函数：
def _select_relevant_verbs(self, object_class):
        """根据物体类型选择相关动词"""
        # 基于物体类型选择可能的动词
        verb_groups = {
            'vehicle': ['ride', 'drive', 'board', 'exit', 'push', 'wash', 'park', 'repair'],
            'food': ['eat', 'cook', 'cut', 'serve', 'smell', 'hold', 'make'],
            'furniture': ['sit_on', 'lie_on', 'stand_on', 'move', 'clean', 'push'],
            'electronic': ['use', 'type_on', 'watch', 'control', 'operate', 'check'],
            'animal': ['pet', 'feed', 'ride', 'hug', 'walk', 'train', 'groom'],
            'sports': ['throw', 'catch', 'kick', 'hit', 'play', 'hold', 'swing'],
            'tool': ['use', 'hold', 'cut_with', 'repair', 'wield', 'work_with']
        }
        
        # 物体类别映射
        category_map = {
            'bicycle': 'vehicle', 'motorcycle': 'vehicle', 'car': 'vehicle',
            'bus': 'vehicle', 'train': 'vehicle', 'truck': 'vehicle', 'airplane': 'vehicle',
            'apple': 'food', 'banana': 'food', 'pizza': 'food', 'sandwich': 'food',
            'cake': 'food', 'donut': 'food', 'hot_dog': 'food',
            'chair': 'furniture', 'couch': 'furniture', 'bed': 'furniture',
            'dining_table': 'furniture', 'bench': 'furniture', 'toilet': 'furniture',
            'tv': 'electronic', 'laptop': 'electronic', 'cell_phone': 'electronic',
            'keyboard': 'electronic', 'remote': 'electronic', 'mouse': 'electronic',
            'dog': 'animal', 'cat': 'animal', 'horse': 'animal', 'cow': 'animal',
            'sheep': 'animal', 'bird': 'animal', 'elephant': 'animal',
            'sports_ball': 'sports', 'frisbee': 'sports', 'tennis_racket': 'sports',
            'baseball_bat': 'sports', 'skateboard': 'sports', 'surfboard': 'sports',
            'knife': 'tool', 'scissors': 'tool', 'fork': 'tool', 'spoon': 'tool'
        }
可以发现category_map并没有包括80个所有的物体类别，你帮我补充完整。

3. _predict_interactions_with_clip函数的选取最相关的几个动作的代码如下：
# 构建文本提示
        text_prompts = [f"a person {verb} a {object_class}" for verb in relevant_verbs[:15]]
        
        # CLIP推理
        try:
            with torch.no_grad():
                inputs = self.clip_processor(
                    text=text_prompts,
                    images=interaction_region,
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                ).to(self.device)
                
                outputs = self.clip_model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1).cpu().numpy()[0]
            
            # 返回top-k动词
            top_k = 5
            top_indices = np.argsort(probs)[-top_k:][::-1]
            results = [(relevant_verbs[i], float(probs[i])) 
                      for i in top_indices if i < len(relevant_verbs)]
目前使用的是CLIP模型，但是我发现CLIP模型相当的不好用，你帮我换个BLIP模型吧，来完成可能得动作选择。

以上便是我想让你修改的内容，你需要仔细的读一下hio_multi_agent_systemv6.py，并完成我想让你帮我完善的功能。
