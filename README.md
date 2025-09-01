# 🚗 Real-Time Vehicle Accident Detection using YOLO

This project demonstrates **real-time vehicle accident detection** using the YOLO (You Only Look Once) object detection model.  
It runs YOLO inference on a video file, resizes frames for faster processing, and displays the annotated results.

---

## 📌 Features
- Real-time vehicle detection and accident recognition  
- Works on video files and can be adapted for webcam input  
- Lightweight: resizes frames for faster performance  
- Uses a custom-trained YOLO model (`vehiclee.pt`)  

---

## 🛠️ Requirements
Install the required dependencies:

# bash
pip install ultralytics opencv-python numpy torch torchvision
🚀 Usage
Place your YOLO model file in the repo root and name it vehiclee.pt.

Place your video file (e.g., vec.MOV) in the same folder.

Run the detection script:

bash
Copy code
python detect.py
Controls
Press q to quit the video window.

📂 Project Structure
graphql
Copy code
.
├── detect.py        # Main script for detection
├── vehiclee.pt      # Trained YOLO model (not included in repo due to size)
├── vec.MOV          # Example input video (optional)
└── README.md
📊 Example Output
The script opens a window showing the detection results frame by frame.

(Add screenshots or demo GIF here 👇)

Copy code
results/
 ├── example1.jpg
 └── demo.gif
📜 License
This project is licensed under the MIT License - feel free to use and modify it.

✨ Author
Muhammad Amin
🔗 🔗 [GitHub](https://github.com/aminrustmani) | [LinkedIn](linkedin.com/in/muhammadamin5214)

