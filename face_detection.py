import cv2
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import numpy as np
import json
import threading
import time

class FaceDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Face Detection App")
        self.root.geometry("800x900")
        
        self.style = ttk.Style()
        self.style.theme_use("clam")
        
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        self.image_tab = ttk.Frame(self.notebook)
        self.video_tab = ttk.Frame(self.notebook)
        self.settings_tab = ttk.Frame(self.notebook)
        
        self.notebook.add(self.image_tab, text="Image Processing")
        self.notebook.add(self.video_tab, text="Video Processing")
        self.notebook.add(self.settings_tab, text="Settings")
        
        self.setup_image_tab()
        self.setup_video_tab()
        self.setup_settings_tab()
        
        self.image = None
        self.video_capture = None
        self.is_processing_video = False
        
        self.cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        self.load_settings()  # Fixed: Added parentheses to call the method
        
    def setup_image_tab(self):
        self.upload_button = ttk.Button(self.image_tab, text="Upload Image", command=self.upload_image)
        self.upload_button.pack(pady=10)
        
        self.detect_button = ttk.Button(self.image_tab, text="Detect Image", command=self.detect_faces, state=tk.DISABLED)
        self.detect_button.pack(pady=10)
        
        self.save_button = ttk.Button(self.image_tab, text="Save Image", command=self.save_image, state=tk.DISABLED)
        self.save_button.pack(pady=10)
        
        self.display_label = ttk.Label(self.image_tab)
        self.display_label.pack(expand=True)
        
    def setup_video_tab(self):
        source_frame = ttk.Frame(self.video_tab)
        source_frame.pack(pady=10)
        
        ttk.Label(source_frame, text="Video Source:").pack(side=tk.LEFT, padx=5)
        self.video_source = tk.StringVar(value="0")
        self.video_source_entry = ttk.Entry(source_frame, textvariable=self.video_source, width=10)
        self.video_source_entry.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(source_frame, text="(0 for default camera, file path for video file)").pack(side=tk.LEFT, padx=5)
        
        self.start_video_button = ttk.Button(self.video_tab, text="Start Video Processing", command=self.start_video_processing)
        self.start_video_button.pack(pady=10)
        
        self.stop_video_button = ttk.Button(self.video_tab, text="Stop Video Processing", command=self.stop_video_processing,
                                            state=tk.DISABLED)
        self.stop_video_button.pack(pady=10)
        
        self.video_label = ttk.Label(self.video_tab)
        self.video_label.pack(expand=True)
        
    def setup_settings_tab(self):
        self.scale_factor = tk.DoubleVar(value=1.1)
        self.min_neighbors = tk.IntVar(value=5)
        self.min_size = tk.IntVar(value=30)
        
        ttk.Label(self.settings_tab, text="Scale Factor:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        ttk.Entry(self.settings_tab, textvariable=self.scale_factor).grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(self.settings_tab, text="Min Neighbors:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        ttk.Entry(self.settings_tab, textvariable=self.min_neighbors).grid(row=1, column=1, padx=5, pady=5)
        
        ttk.Label(self.settings_tab, text="Min Size:").grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
        ttk.Entry(self.settings_tab, textvariable=self.min_size).grid(row=2, column=1, padx=5, pady=5)
        
        ttk.Button(self.settings_tab, text="Save Settings", command=self.save_settings).grid(row=3, column=0,
                                                                                             columnspan=2,
                                                                                             padx=5, pady=5)
        
    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.png *.jpeg")])
        if file_path:
            self.image = cv2.imread(file_path)
            self.show_image(self.image)
            self.detect_button.config(state=tk.NORMAL)
            
    def detect_faces(self):
        if self.image is None:
            messagebox.showwarning("Warning", "Please upload an image first")
            return
        
        # Create a copy of the image to draw on
        image_copy = self.image.copy()
        gray = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)
        faces = self.cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor.get(),
            minNeighbors=self.min_neighbors.get(),
            minSize=(self.min_size.get(), self.min_size.get())
        )
        
        for (x, y, w, h) in faces:
            cv2.rectangle(image_copy, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
        self.show_image(image_copy)
        self.save_button.config(state=tk.NORMAL)
        
    def show_image(self, cv_img):
        # Resize image to fit in the window if it's too large
        height, width = cv_img.shape[:2]
        max_size = 600
        if height > max_size or width > max_size:
            scale = max_size / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            cv_img = cv2.resize(cv_img, (new_width, new_height))
        
        cv_img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(cv_img_rgb)
        img_tk = ImageTk.PhotoImage(image=img_pil)
        self.display_label.imgtk = img_tk
        self.display_label.configure(image=img_tk)
        
    def save_image(self):
        if self.image is None:
            messagebox.showwarning("Warning", "No processed image to save")
            return
        
        file_path = filedialog.asksaveasfilename(defaultextension=".jpg", 
                                                filetypes=[("JPEG files", "*.jpg"), 
                                                          ("PNG files", "*.png"),
                                                          ("All files", "*.*")])
        if file_path:
            cv2.imwrite(file_path, self.image)
            messagebox.showinfo("Success", "Image saved successfully")
            
    def start_video_processing(self):
        source = self.video_source.get()
        
        # Try to convert to integer if it's a number
        try:
            source = int(source)
        except ValueError:
            pass  # It's a file path, keep as string
        
        self.video_capture = cv2.VideoCapture(source)
        if not self.video_capture.isOpened():
            messagebox.showerror("Error", "Could not open video source")
            return
        
        self.is_processing_video = True
        self.start_video_button.config(state=tk.DISABLED)
        self.stop_video_button.config(state=tk.NORMAL)
        
        # Start video processing in a separate thread
        self.video_thread = threading.Thread(target=self.process_video, daemon=True)
        self.video_thread.start()
        
    def stop_video_processing(self):
        self.is_processing_video = False
        if self.video_capture:
            self.video_capture.release()
        self.start_video_button.config(state=tk.NORMAL)
        self.stop_video_button.config(state=tk.DISABLED)
        self.video_label.configure(image='')
        
    def process_video(self):
        while self.is_processing_video:
            ret, frame = self.video_capture.read()
            if not ret:
                # If we can't read a frame, try to reopen the capture
                try:
                    source = self.video_source.get()
                    try:
                        source = int(source)
                    except ValueError:
                        pass
                    self.video_capture = cv2.VideoCapture(source)
                    if not self.video_capture.isOpened():
                        break
                    continue
                except:
                    break
            
            # Process the frame for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.cascade.detectMultiScale(
                gray,
                scaleFactor=self.scale_factor.get(),
                minNeighbors=self.min_neighbors.get(),
                minSize=(self.min_size.get(), self.min_size.get())
            )
            
            # Draw rectangles around detected faces
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                
            # Update the video display in the main thread
            self.root.after(0, lambda: self.show_video_frame(frame))
            
            # Control the processing speed
            time.sleep(0.03)  # ~30 FPS
            
        self.video_capture.release()
        
    def show_video_frame(self, frame):
        # Resize frame to fit in the window
        height, width = frame.shape[:2]
        max_size = 600
        if height > max_size or width > max_size:
            scale = max_size / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            frame = cv2.resize(frame, (new_width, new_height))
            
        cv_img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(cv_img_rgb)
        img_tk = ImageTk.PhotoImage(image=img_pil)
        self.video_label.imgtk = img_tk
        self.video_label.configure(image=img_tk)
        
    def save_settings(self):
        settings = {
            "scale_factor": self.scale_factor.get(),
            "min_neighbors": self.min_neighbors.get(),
            "min_size": self.min_size.get(),
        }
        
        with open("face_detection_settings.json", "w") as f:
            json.dump(settings, f)
            
        messagebox.showinfo("Success", "Settings saved successfully")
        
    def load_settings(self):
        try:
            with open("face_detection_settings.json", "r") as f:
                settings = json.load(f)
                
            self.scale_factor.set(settings.get("scale_factor", 1.1))
            self.min_neighbors.set(settings.get("min_neighbors", 5))
            self.min_size.set(settings.get("min_size", 30))
        except FileNotFoundError:
            pass  # Use default settings if file doesn't exist
        
def main():
    root = tk.Tk()
    app = FaceDetectionApp(root)
    root.mainloop()
    
if __name__ == "__main__":
    main()