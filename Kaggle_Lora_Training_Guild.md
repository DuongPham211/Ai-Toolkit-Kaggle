Hướng Dẫn Chạy AI-Toolkit Trên Kaggle (Fix Lỗi Meta Tensor & Read-Only)
Tài liệu này tổng hợp các vấn đề gặp phải khi chạy ai-toolkit trên Kaggle với các model lớn (như Flux, Z-Image) và cách khắc phục triệt để.
1. Nguyên nhân gốc rễ
Môi trường Kaggle có đặc thù:
RAM hệ thống (CPU RAM): Khá lớn (~30GB), đủ để load model ban đầu.
VRAM (GPU): Giới hạn (16GB cho T4 hoặc P100), buộc phải dùng Quantization (nén model).
Hệ thống file: Thư mục /kaggle/input là Read-Only (chỉ đọc), không thể ghi file cache vào đó.
Meta Tensor: Thư viện accelerate thường cố gắng tối ưu bằng cách load model vào thiết bị ảo "Meta" (không có dữ liệu thật). Điều này gây lỗi khi ta cố gắng Merge LoRA hoặc Quantize vì không có dữ liệu để tính toán.
2. Cấu hình config.yaml
Cần điều chỉnh file config để ép buộc hệ thống tải model vào RAM thật và kích hoạt nén.
code
Yaml
model:
  # ... các thông số khác ...
  quantize: true           # BẮT BUỘC: Để nén model cho vừa VRAM T4
  qtype: "qfloat8"         # Định dạng nén
  
  # QUAN TRỌNG: Tắt các chế độ tiết kiệm RAM ảo để tránh lỗi Meta Tensor
  low_vram: false          
  
  # Thêm tham số này để ép thư viện load model vào CPU RAM thật
  model_kwargs:
    low_cpu_mem_usage: false
    device_map: null       # Không để accelerate tự chia device
3. Các File Code Cần Patch (Vá Lỗi)
Bạn cần ghi đè hoặc sửa đổi các file mã nguồn của ai-toolkit để tương thích với Kaggle.
A. extensions_built_in/diffusion_models/z_image/z_image.py
Vấn đề: Model được load ở chế độ Meta/Lazy, gây crash khi chuyển sang GPU hoặc khi merge LoRA.
Giải pháp:
Hard-code tham số low_cpu_mem_usage=False trong hàm from_pretrained.
Thêm cơ chế kiểm tra an toàn (Safe-guard) trước khi gọi lệnh .to(device).
Đoạn code sửa đổi chính:
code
Python
# Trong hàm load_model()
transformer = ZImageTransformer2DModel.from_pretrained(
    transformer_path, 
    subfolder=transformer_subfolder, 
    torch_dtype=dtype,
    low_cpu_mem_usage=False,  # <--- QUAN TRỌNG: Ép load thật
    device_map=None,
    ignore_mismatched_sizes=True
)

# ... (đoạn xử lý khác) ...

# Kiểm tra an toàn khi di chuyển model
try:
    is_meta = any(p.device.type == 'meta' for p in pipe.transformer.parameters())
    if not is_meta:
        pipe.transformer = pipe.transformer.to(self.device_torch)
    else:
        print("⚠️ Warning: Transformer is on Meta device. Skipping .to() move.")
except Exception as e:
    print(f"⚠️ Warning: Could not move transformer to device: {e}")
B. toolkit/util/quantize.py
Vấn đề: Khi nén (Quantize), nếu gặp Meta Tensor sẽ gây lỗi NotImplementedError. Cũng cần fix lỗi import get_qtype.
Giải pháp: Thêm kiểm tra if is_meta: continue để bỏ qua các block bị lỗi thay vì crash chương trình.
Đoạn code sửa đổi chính:
code
Python
for block in tqdm(all_blocks):
    # --- FIX: CHECK META TENSOR BEFORE MOVING ---
    is_meta = any(p.device.type == 'meta' for p in block.parameters())
    if is_meta:
        continue # Bỏ qua nếu là meta tensor
    # --------------------------------------------
    
    block.to(base_model.device_torch, dtype=base_model.torch_dtype, non_blocking=True)
    quantize(block, weights=quantization_type)
    # ...
C. toolkit/network_mixins.py
Vấn đề: Hàm merge_in cố gắng cộng trọng số LoRA vào Meta Tensor của model gốc.
Giải pháp: Kiểm tra nếu trọng số gốc là Meta thì bỏ qua việc merge (vì model sẽ được load lại hoặc quantize sau).
Đoạn code sửa đổi chính:
code
Python
# Trong hàm merge_in()
if org_sd[weight_key].device.type == 'meta':
    return # Không thể merge vào meta tensor, return luôn.
D. toolkit/data_loader.py
Vấn đề: ai-toolkit cố gắng ghi file .aitk_size.json vào thư mục dataset (nằm trong /kaggle/input), nơi chỉ có quyền Đọc (Read-only).
Giải pháp: Bọc lệnh ghi file trong khối try...except.
Đoạn code sửa đổi chính:
code
Python
# Trong hàm __init__ của AiToolkitDataset
try:
    with open(dataset_size_file, 'w') as f:
        json.dump(self.size_database, f)
except OSError:
    pass # Bỏ qua lỗi Read-only filesystem trên Kaggle
except Exception as e:
    print_acc(f"Warning: Could not save size database: {e}")
4. Tải Model Thủ Công (Fix lỗi config Text Encoder)
Do cấu trúc thư mục trên HuggingFace của Z-Image-Turbo không chuẩn, AutoTokenizer không tự tìm thấy file config. Cần tải thủ công về đúng thư mục.
Script Python cần chạy trước khi train:
code
Python
from huggingface_hub import hf_hub_download, snapshot_download

repo_id = "Tongyi-MAI/Z-Image-Turbo"
base_dir = "/kaggle/working/Z-Image-Turbo-HF"

# 1. Tải thư mục tokenizer
snapshot_download(repo_id=repo_id, allow_patterns=["tokenizer/*"], local_dir=base_dir, local_dir_use_symlinks=False)

# 2. Tải config cho text_encoder
text_encoder_files = ["text_encoder/config.json", "text_encoder/generation_config.json", "text_encoder/model.safetensors.index.json"]
for file_path in text_encoder_files:
    hf_hub_download(repo_id=repo_id, filename=file_path, local_dir=base_dir, local_dir_use_symlinks=False)
5. Quy Trình Chạy Chuẩn
Để đảm bảo thành công, hãy thực hiện theo thứ tự:
Cài đặt môi trường: Cài ai-toolkit và các thư viện cần thiết.
Tải Model: Chạy script ở mục 4 để tải Z-Image và các file config thiếu.
Patch Code: Chạy các cell Python (đã cung cấp ở các bước trước) để ghi đè các file z_image.py, quantize.py, network_mixins.py, data_loader.py.
Cấu hình Config: Tạo file config.yaml với các thiết lập ở mục 2.
Start Training: Chạy lệnh python run.py config.yaml.

config.yaml for Z Image
config_yaml = r"""
job: "extension"

config:
  name: "testarossa_zimage_lora"
  
  process:
    - type: "diffusion_trainer"
      training_folder: "/kaggle/working/testarossa_lora"
      sqlite_db_path: "./aitk_db.db"
      device: "cuda"
      trigger_word: "testarossa"
      performance_log_every: 10
      
      network:
        type: "lora"
        linear: 32
        linear_alpha: 32
        conv: 0
        conv_alpha: 0
        lokr_full_rank: false
        lokr_factor: -1
        network_kwargs:
          ignore_if_contains: []
      
      save:
        dtype: "bf16"
        save_every: 300
        max_step_saves_to_keep: 10
        save_format: "safetensors"
        push_to_hub: false
      
      datasets:
        - folder_path: "/kaggle/input/testarossa-dataset/testarossa-dataset"
          mask_path: null
          mask_min_value: 0.1
          default_caption: "testarossa a woman"
          caption_ext: "txt"
          caption_dropout_rate: 0.05
          cache_latents_to_disk: false
          is_reg: false
          network_weight: 1
          resolution: [768, 1024]
          controls: []
          shrink_video_to_frames: true
          num_frames: 1
          do_i2v: false
          flip_x: false
          flip_y: false
      
      train:
        batch_size: 1
        bypass_guidance_embedding: false
        steps: 2000
        gradient_accumulation: 1
        train_unet: true
        train_text_encoder: false
        gradient_checkpointing: true
        noise_scheduler: "flowmatch"
        optimizer: "adamw8bit"
        timestep_type: "sigmoid"
        content_or_style: "balanced"
        optimizer_params:
          weight_decay: 0.0001
        unload_text_encoder: false
        cache_text_embeddings: true
        lr: 0.00015
        ema_config:
          use_ema: false
          ema_decay: 0.99
        skip_first_sample: false
        force_first_sample: false
        disable_sampling: false
        dtype: "bf16"
        diff_output_preservation: false
        diff_output_preservation_multiplier: 1
        diff_output_preservation_class: "person"
        switch_boundary_every: 1
        loss_type: "mse"
        do_differential_guidance: true
        differential_guidance_scale: 3
      
      model:
        name_or_path: "/kaggle/working/Z-Image-Turbo-HF"
        quantize: true
        qtype: "qfloat8"
        quantize_te: true
        qtype_te: "qfloat8"
        arch: "zimage:turbo"
        low_vram: false
        layer_offloading: false
        layer_offloading_text_encoder_percent: 1
        layer_offloading_transformer_percent: 1
        assistant_lora_path: "/kaggle/working/zimage_turbo_training_adapter/zimage_turbo_training_adapter_v2.safetensors"
        # Bỏ device_map để tránh xung đột, tin tưởng vào patch z_image.py
        model_kwargs:
          low_cpu_mem_usage: false
      
      sample:
        sampler: "flowmatch"
        sample_every: 300
        width: 1024
        height: 1024
        samples:
          - prompt: "testarossa a woman, cinematic portrait, high detail"
          - prompt: "testarossa a woman in red dress, studio lighting"
        neg: ""
        seed: 42
        walk_seed: true
        guidance_scale: 1
        sample_steps: 8
        num_frames: 1
        fps: 1

meta:
  name: "testarossa_zimage_lora"
  version: "1.0"
"""

with open("/kaggle/working/ai-toolkit/config.yaml", "w") as f:
    f.write(config_yaml)

print("✅ Config Updated. Hãy chạy lại lệnh train.")