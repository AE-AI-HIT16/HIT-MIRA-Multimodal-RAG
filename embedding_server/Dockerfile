# Ảnh cho pod GPU trên RunPod. Tách hẳn khỏi `runpod_worker/`: cái kia là job
# hàng đợi, chạy xong là chết, cố tình không có credential nào; cái này là
# dịch vụ sống lâu, mang theo 3,5GB trọng số. Gộp chung thành một ảnh nghĩa là
# mỗi job xuất artifact cũng phải kéo theo cả đống torch + trọng số.
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

WORKDIR /app

# Cài dependency trước, copy code sau: sửa code thì không phải build lại lớp
# nặng nhất.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Nướng trọng số vào ảnh. Tải lúc khởi động thì mỗi lần pod restart lại mất
# vài phút và phụ thuộc vào HuggingFace còn sống — đúng kiểu phụ thuộc bên
# ngoài mà việc tự host này sinh ra để loại bỏ.
ENV HF_HOME=/opt/hf
RUN python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('jinaai/jina-clip-v2', allow_patterns=['*.json','*.py','*.txt','*.model','*.safetensors'])"

COPY encoder.py server.py ./

# Trên GPU thì nạp sẵn: pod sống lâu nên trả cái giá 30 giây một lần lúc khởi
# động, đổi lấy việc request đầu tiên không bị treo.
ENV EMBED_PRELOAD=1
EXPOSE 8100

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8100"]
