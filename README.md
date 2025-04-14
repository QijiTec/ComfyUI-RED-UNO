# ComfyUI-RED-UNO
Default 16GB VRAM UNO in context generation ComfyUI-node, using RED-UNO FT model

默认16G显存的UNO ComfyUI in-context生成组件，使用RED-UNO FT模型

## RED-UNO FLUX-dev-FT FP8：

https://civitai.com/models/958009/redcraft-or-cads-or-updated-apr14-or-commercial-and-advertising-design-system

![image](https://github.com/user-attachments/assets/d354655e-a446-48f3-893b-edfb5e61d6f7)


## RED. UNO In-Context (FP8) 4/14/2025

REDAIGC FT Model used to match UNO In-Context Generation

(with improved quality compared to F.1 dev)

解决了FLUX FT底模无法适配UNO组件的问题，FP8权重（显存占用16GB），同时支持Diffusers以及ComfyUI

实测对比DEV生成质量明显提升，对比BF16版本没有明显质量损失，显存占用只有16G，生成时间大概30秒

# 注意：16G显存占用指的是官方案例中512x512尺寸的生成，640x960以上显存占用暴涨，目前依旧建议使用Diffusers推理环境

# 安装说明

由于ComfyUI-RED-UNO使用Diffusers库进行推理，所以首次使用，会自动下载Diffusers格式的T5和Clip（10G左右）

存放路径是ComfyUI虚拟环境的.cache缓存文件夹，如果是系统级安装的ComfyUI，则会存放至C盘User路径下的.cache（非常占空间）

建议是给ComfyUI的虚拟环境指定一个独立的缓存目录。

FLUX FT底模正常存放在ComfyUI\models\diffusion_models目录下，VAE在ComfyUI\models\vae，UNO-Lora在ComfyUI\models\loras

![image](https://github.com/user-attachments/assets/572a5206-ab23-417f-b83e-61b2f06d0b8b)

use_fp8和offload 默认勾选，否则24G显存就不够用了，VAE也要使用Diffusers格式的版本否则会报错（ComfyUI目前没有直接支持UNO所以才这么折腾）

VAE版本：
https://huggingface.co/diffusers/FLUX.1-vae/tree/main


---

Diffusers 脚本：

https://github.com/bytedance/UNO

Dit-LoRA 权重：

https://huggingface.co/bytedance-research/UNO

VAE版本：

https://huggingface.co/diffusers/FLUX.1-vae/tree/main


# ComfyUI UNO Nodes

ComfyUI UNO Nodes is a collection of nodes for ComfyUI that allows you to load and use UNO models.

[https://github.com/bytedance/UNO comfyui](https://github.com/QijiTec/ComfyUI-RED-UNO.git)


FP8 support
open offload and fp8 support 16GB VRAM

flux model in unet directory and lora in lora directory

clip and t5 will autodownload


![UNO0](https://github.com/user-attachments/assets/2d5e287e-73fc-4e95-ba3d-42f81ef920fc)



## Paper Analysis

**UNO - Less-to-More Generalization Unlocking More Controllability by In-Context Generation**

**Introduction**

Subject-driven image generation, aiming to create new images based on textual descriptions and user-provided reference images, is a central challenge in the field of Artificial Intelligence Generated Content (AIGC). However, existing methods still face significant limitations, particularly regarding **data scalability** (especially acquiring high-quality multi-subject paired data) and **subject expansibility** (stably and controllably handling multiple subjects). Researchers from the Intelligent Creation Team at ByteDance address these issues in their paper, "Less-to-More Generalization: Unlocking More Controllability by In-Context Generation," by proposing a novel framework named **UNO (Universal aNd cOntrollable)**, based on an innovative "Less-to-More" generalization paradigm.

**Core Challenges and Motivation**

Traditional approaches grapple with data scarcity (especially for multi-subject pairs), difficulties in multi-subject control (identity confusion, layout issues), and the inherent trade-off between efficiency and fidelity. UNO's motivation stems from the need to systematically tackle these bottlenecks.

**UNO's Core Contributions and Technical Breakdown**

UNO presents a comprehensive solution encompassing data, model, and training strategies:

1.  **Innovative "Model-Data Co-evolution" Paradigm:** This is UNO's most groundbreaking conceptual contribution. Instead of passively relying on existing data, it proposes a virtuous cycle where the model itself (even less capable versions) is leveraged to systematically synthesize higher-quality, more complex customized data (evolving from single-subject to multi-subject). This superior data, in turn, trains more powerful ("more controllable") model variants.
2.  **High-Quality Synthetic Data Automation Pipeline:** Utilizes the in-context generation capability of Diffusion Transformers (DiTs), high-resolution output, and a sophisticated multi-stage filtering process involving DINOv2 + VLM (with Chain-of-Thought prompting) to automatically generate large-scale, high-fidelity, high-consistency single- and multi-subject paired data.
3.  **UNO Model Architecture and Training Strategy:** Built upon a DiT architecture, it employs **Progressive Cross-Modal Alignment** (training first on single-subject, then multi-subject data) and introduces the crucial component: **UnoPE**.

**In-Depth Comparison: UNO vs. OminiControl vs. In-Context LoRA**

All three approaches recognize and leverage the powerful in-context learning capabilities of Diffusion Transformers (DiTs), but their methodologies and focuses differ significantly. Understanding these distinctions highlights UNO's unique value:

*   **Core Mechanism Utilization:**
    *   **OminiControl:** Was among the earlier works demonstrating that DiTs can understand and replicate reference subjects through specific input formats (like side-by-side image templates) without explicit fine-tuning. It primarily relies on this "emergent" capability of the DiT itself.
    *   **In-Context LoRA:** Focuses on using the DiT's context window for "few-shot learning" or "rapid adaptation." It posits that by providing a few (reference, target) example pairs within the context, the model can quickly grasp a new concept. This rapid adaptation is then potentially enhanced or guided by training specific LoRA modules designed to process this contextual information.
    *   **UNO:** Does *not* rely on immediate contextual examples for learning. Instead, it utilizes large-scale **pre-synthesized**, high-quality data for **pre-training/fine-tuning**. This process internalizes the ability to understand and handle reference images (both single and multiple) directly into the model's weights (or attached modules like LoRA). It's more akin to "compiling" the context understanding capability into the model beforehand.

*   **Data Strategy Differences:**
    *   **OminiControl:** Data generation appears relatively basic, potentially at lower resolutions, and with less sophisticated filtering.
    *   **In-Context LoRA:** Likely relies on training with collections of high-quality (reference, target) example pairs.
    *   **UNO:** Employs the most complex and systematic data strategy. It emphasizes high resolution, multiple aspect ratios, progressive generation from single to multi-subject, and rigorous filtering using VLM CoT. **Data quality is explicitly treated as crucial for pushing the model's performance ceiling**, embodying the "model-data co-evolution" philosophy.

*   **Multi-Subject Handling Capability:**
    *   **OminiControl:** Lacks mechanisms explicitly designed for multi-subject scenarios. Its multi-subject performance likely depends heavily on the DiT's raw generalization ability, which might struggle with complex interactions or attribute distinction.
    *   **In-Context LoRA:** The original paper doesn't heavily focus on specific solutions for multi-subject generation. Its effectiveness might be limited by the complexity of multi-subject examples provided in the context and the base DiT's generalization limits.
    *   **UNO:** This is a standout advantage for UNO. It not only explicitly includes multi-subject scenarios in its data generation and training phases but, critically, introduces **UnoPE (Universal Rotary Position Embedding)**. UnoPE modifies positional encodings by assigning unique offsets to tokens from different reference images. This fundamentally helps the Transformer's attention mechanism **distinguish between different visual sources**, significantly **mitigating attribute confusion** between multiple subjects. It also encourages the model to prioritize the text prompt for layout instructions rather than simply replicating the spatial arrangement of reference images, leading to more robust and controllable multi-subject generation.

*   **Training and Inference:**
    *   **OminiControl:** Training is likely simpler; inference requires providing input matching the expected template.
    *   **In-Context LoRA:** Requires training specific LoRA modules. Inference might necessitate providing contextual example pairs along with loading the appropriate LoRA weights.
    *   **UNO:** The training pipeline (including data generation and progressive stages) is more complex. However, the **inference-time user experience is simpler and aligns with standard tuning-free methods** like IP-Adapter. Users only need to provide the reference image(s) and the text prompt.

**Comparison Summary (UNO vs. OminiControl vs. In-Context LoRA):**
OminiControl represents an early exploration of DiT's in-context capabilities. In-Context LoRA investigates rapid concept adaptation using context, often coupled with LoRA. **UNO, however, offers a more comprehensive and systematic solution.** It leverages DiT's context abilities but significantly enhances the baseline capability through a superior data strategy (co-evolution, synthesis, filtering). Crucially, it **addresses the multi-subject challenge head-on with the dedicated UnoPE mechanism**, leading to robust and controllable generation in complex scenarios, all while maintaining a user-friendly, tuning-free inference process.

**Experimental Results and Performance**

UNO's effectiveness is strongly supported by its experimental results:

*   **Quantitative SOTA:** Achieves state-of-the-art scores on DreamBench (single-subject) and multi-subject benchmarks for subject similarity metrics (DINO, CLIP-I), while maintaining highly competitive text fidelity (CLIP-T).
*   **Qualitative Excellence:** Produces high-quality, natural-looking images. Demonstrates strong subject identity preservation, detail fidelity, stable multi-subject control, and effective attribute editing capabilities.
*   **Strong Generalization:** Showcases impressive performance across diverse applications like virtual try-on, product design, identity preservation, and stylized generation.
*   **Ablation Studies:** Thoroughly validate the necessity and effectiveness of each component: high-quality synthetic data, progressive training, and especially the UnoPE mechanism.

**Conclusion**

The UNO framework marks a significant advancement in controllable image generation. Through its unique "model-data co-evolution" concept, a superior synthetic data pipeline, and a well-designed model architecture featuring UnoPE and progressive training, UNO successfully overcomes critical bottlenecks in data scalability and multi-subject control. It not only achieves state-of-the-art performance but, more importantly, demonstrates a viable path towards unlocking greater controllability and generalization via systematic model self-improvement. As a powerful, unified, and tuning-free solution, UNO offers substantial technological support for personalized content creation, virtual reality, design, and beyond.

Future work could further enhance UNO's versatility by expanding the types of synthetic data generated (e.g., including more editing or stylization pairs) to cover an even broader range of applications.

---
