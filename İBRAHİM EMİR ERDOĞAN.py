import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
import os

class ImageProcessorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Görüntü İşleme Arayüzü")
        self.root.geometry("950x600")
        
        # Tema renkleri
        self.bg_color = "#1a237e"  # Koyu lacivert
        self.button_bg = "#3949ab"  # Buton arka plan rengi
        self.button_fg = "#ffffff"  # Buton yazı rengi
        self.button_active_bg = "#5c6bc0"  # Buton aktif rengi
        self.slider_bg = "#3949ab"  # Slider arka plan rengi
        self.slider_fg = "#ffffff"  # Slider ön plan rengi
        
        # Ana pencere arka plan rengi
        self.root.configure(bg=self.bg_color)
        
        self.image = None
        self.original_image = None
        self.display_image = None

        # Ana frame
        main_frame = tk.Frame(root, bg=self.bg_color)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=0, pady=10)

        # Sol panel (butonlar için)
        left_panel = tk.Frame(main_frame, width=420, bg=self.bg_color)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=0, pady=5)
        left_panel.pack_propagate(False)  # Genişliği sabit tut
        left_panel.grid_propagate(False)

        # Canvas ve Scrollbar için frame
        canvas_frame = tk.Frame(left_panel, bg=self.bg_color)
        canvas_frame.pack(fill=tk.BOTH, expand=True)

        # Canvas oluştur
        canvas = tk.Canvas(canvas_frame, bg=self.bg_color, highlightthickness=0)
        scrollbar = ttk.Scrollbar(canvas_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg=self.bg_color)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Canvas ve scrollbar'ı yerleştir
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Butonları scrollable frame'e ekle
        buttons = [
            ("Görsel Yükle", self.load_image),
            ("Resmi Sıfırla", self.reset_image),
            ("Griye Çevir", self.to_gray),
            ("Kaydet", self.save_image),
            ("RGB Kanallarını Göster", self.show_rgb_channels),
            ("Negatifini Al", self.negative_image),
            ("Histogram Göster", self.show_histogram),
            ("Histogram Eşitle", self.equalize_histogram),
            ("Taşı", self.translate_image),
            ("Aynala", self.mirror_image),
            ("Eğ (Shear)", self.shear_image),
            ("Yakınlaştır", lambda: self.scale_image(1.2)),
            ("Uzaklaştır", lambda: self.scale_image(0.8)),
            ("Döndür (90°)", self.rotate_image),
            ("Kırp (Orta)", self.crop_image),
            ("Ortalama Filtre", self.mean_filter),
            ("Medyan Filtre", self.median_filter),
            ("Gauss Filtre", self.gauss_filter),
            ("Konservatif Filtre", self.conservative_filter),
            ("Crimmins Speckle", self.crimmins_speckle_filter),
            ("Fourier Spektrum Göster", self.show_fourier),
            ("Low Pass Filter", self.low_pass_filter),
            ("High Pass Filter", self.high_pass_filter),
            ("Band Geçiren Filtre", self.band_pass_filter),
            ("Band Durduran Filtre", self.band_stop_filter),
            ("Butterworth LPF", self.butterworth_lpf),
            ("Butterworth HPF", self.butterworth_hpf),
            ("Gaussian LPF", self.gaussian_lpf),
            ("Gaussian HPF", self.gaussian_hpf),
            ("Homomorfik Filtre", self.homomorphic_filter),
            ("Sobel", self.sobel_edge),
            ("Prewitt X", self.prewitt_x_edge),
            ("Prewitt Y", self.prewitt_y_edge),
            ("Prewitt Hepsi", self.prewitt_all_edge),
            ("Roberts Cross", self.roberts_edge),
            ("Compass (Kirsch)", self.compass_edge),
            ("Canny", self.canny_edge),
            ("Laplace", self.laplace_edge),
            ("Gabor", self.gabor_filter),
            ("Hough Çizgi", self.hough_lines),
            ("k-means Segmentasyon", self.kmeans_segmentation),
            ("Erode", self.erode_image),
            ("Dilate", self.dilate_image)
        ]

        # Butonları grid ile 3'lü gruplar halinde yerleştir
        for idx, (text, command) in enumerate(buttons):
            row = idx // 3
            col = idx % 3
            btn = tk.Button(
                scrollable_frame,
                text=text,
                command=command,
                width=13,
                bg=self.button_bg,
                fg=self.button_fg,
                activebackground=self.button_active_bg,
                activeforeground=self.button_fg,
                relief=tk.FLAT,
                padx=3,
                pady=3,
                font=("Segoe UI", 9)
            )
            btn.grid(row=row, column=col, padx=2, pady=2, sticky="nsew")
            # Hover efekti
            btn.bind("<Enter>", lambda e, b=btn: b.configure(bg=self.button_active_bg))
            btn.bind("<Leave>", lambda e, b=btn: b.configure(bg=self.button_bg))
        for i in range(3):
            scrollable_frame.grid_columnconfigure(i, weight=1)

        # Sliderları scrollable frame'e ekle
        slider_frame = tk.Frame(scrollable_frame, bg=self.bg_color)
        slider_frame.grid(row=(len(buttons) // 3) + 1, column=0, columnspan=3, pady=10, padx=2, sticky="ew")
        
        # Parlaklık slider'ı
        brightness_label = tk.Label(slider_frame, text="Parlaklık", bg=self.bg_color, fg=self.button_fg, font=("Segoe UI", 9))
        brightness_label.pack(anchor=tk.W)
        self.brightness_slider = ttk.Scale(
            slider_frame,
            from_=-100,
            to=100,
            orient=tk.HORIZONTAL,
            length=180,
            command=self._on_slider_change,
            style="Custom.Horizontal.TScale"
        )
        self.brightness_slider.set(0)
        self.brightness_slider.pack(fill=tk.X, pady=(0, 5))
        
        # Kontrast slider'ı
        contrast_label = tk.Label(slider_frame, text="Kontrast", bg=self.bg_color, fg=self.button_fg, font=("Segoe UI", 9))
        contrast_label.pack(anchor=tk.W)
        self.contrast_slider = ttk.Scale(
            slider_frame,
            from_=-100,
            to=100,
            orient=tk.HORIZONTAL,
            length=180,
            command=self._on_slider_change,
            style="Custom.Horizontal.TScale"
        )
        self.contrast_slider.set(0)
        self.contrast_slider.pack(fill=tk.X)

        # Sağ panel (görüntü için)
        right_panel = tk.Frame(main_frame, bg=self.bg_color, width=400)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=0, pady=5)
        right_panel.pack_propagate(False)
        
        # Görüntü gösterim alanı
        self.panel = tk.Label(right_panel, bg='#2c3e50')  # Koyu gri arka plan
        self.panel.pack(fill=tk.BOTH, expand=True)

        # Ttk stil ayarları
        style = ttk.Style()
        style.configure(
            "Custom.Horizontal.TScale",
            background=self.bg_color,
            troughcolor=self.slider_bg,
            sliderthickness=15,
            sliderlength=20
        )

    def load_image(self):
        try:
            # Desteklenen dosya formatları
            filetypes = [
                ('Image files', '*.jpg *.jpeg *.png *.bmp *.gif *.tiff'),
                ('JPEG files', '*.jpg *.jpeg'),
                ('PNG files', '*.png'),
                ('BMP files', '*.bmp'),
                ('All files', '*.*')
            ]
            
            file_path = filedialog.askopenfilename(
                title="Görüntü Seç",
                filetypes=filetypes
            )
            
            if not file_path:  # Kullanıcı iptal ettiyse
                print("Dosya seçimi iptal edildi")
                return
                
            print(f"Seçilen dosya: {file_path}")
            
            # Dosyanın var olup olmadığını kontrol et
            if not os.path.exists(file_path):
                print(f"Hata: Dosya bulunamadı: {file_path}")
                messagebox.showerror("Hata", f"Dosya bulunamadı:\n{file_path}")
                return
                
            # Dosya boyutunu kontrol et
            file_size = os.path.getsize(file_path)
            if file_size == 0:
                print(f"Hata: Dosya boş: {file_path}")
                messagebox.showerror("Hata", "Seçilen dosya boş!")
                return
                
            print(f"Dosya boyutu: {file_size} bytes")
            
            # Dosya uzantısını kontrol et
            file_ext = os.path.splitext(file_path)[1].lower()
            print(f"Dosya uzantısı: {file_ext}")
            
            # Görüntüyü yükle
            try:
                # Önce PIL ile yüklemeyi dene
                pil_image = Image.open(file_path)
                # PIL görüntüsünü numpy dizisine çevir
                loaded_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                print("Görüntü PIL ile başarıyla yüklendi")
            except Exception as pil_error:
                print(f"PIL ile yükleme başarısız: {str(pil_error)}")
                # PIL başarısız olursa OpenCV ile dene
                try:
                    loaded_image = cv2.imread(file_path, cv2.IMREAD_COLOR)
                    if loaded_image is None:
                        raise Exception("OpenCV görüntüyü yükleyemedi")
                    print("Görüntü OpenCV ile başarıyla yüklendi")
                except Exception as cv_error:
                    print(f"OpenCV ile yükleme başarısız: {str(cv_error)}")
                    raise Exception(f"Görüntü yüklenemedi:\nPIL hatası: {str(pil_error)}\nOpenCV hatası: {str(cv_error)}")
            
            if loaded_image is None:
                raise Exception("Görüntü yüklendi ama None değeri döndü")
            
            # Orijinal görüntüyü sakla
            self.original_image = loaded_image.copy()
            self.image = loaded_image.copy()
                
            print(f"Görüntü başarıyla yüklendi")
            print(f"Görüntü boyutları: {self.image.shape}")
            print(f"Görüntü tipi: {self.image.dtype}")
            print(f"Görüntü değer aralığı: [{self.image.min()}, {self.image.max()}]")
            
            # Görüntüyü göster
            self.show_image(self.image)
            
        except Exception as e:
            error_msg = f"Görüntü yüklenirken hata oluştu:\n{str(e)}"
            print(error_msg)
            messagebox.showerror("Hata", error_msg)
            # Hata durumunda görüntüyü temizle
            self.image = None
            self.original_image = None
            self.panel.config(image='')
            self.panel.image = None

    def reset_image(self):
        """Orijinal görüntüyü geri yükle"""
        if self.original_image is not None:
            self.image = self.original_image.copy()
            self.show_image(self.image)
            # Sliderları sıfırla
            self.brightness_slider.set(0)
            self.contrast_slider.set(0)
            print("Görüntü orijinal haline sıfırlandı")
        else:
            messagebox.showwarning("Uyarı", "Sıfırlanacak orijinal görüntü yok!")

    def show_image(self, img):
        if img is None:
            print("Hata: Görüntü None!")
            return
        try:
            print(f"Görüntü boyutları: {img.shape}")
            # BGR'den RGB'ye çevir
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # Görüntüyü panel boyutuna göre yeniden boyutlandır
            panel_width = self.panel.winfo_width()
            panel_height = self.panel.winfo_height()
            if panel_width > 1 and panel_height > 1:  # Panel boyutları hazırsa
                # En-boy oranını koru
                h, w = img_rgb.shape[:2]
                aspect = w/h
                if w > panel_width:
                    new_w = panel_width
                    new_h = int(new_w/aspect)
                else:
                    new_h = panel_height
                    new_w = int(new_h*aspect)
                img_resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
            else:
                img_resized = img_rgb
            im_pil = Image.fromarray(img_resized)
            imgtk = ImageTk.PhotoImage(image=im_pil)
            self.display_image = imgtk  # Referansı tut
            self.panel.config(image=imgtk)
            self.panel.image = imgtk    # Referansı tut
            print("Görüntü başarıyla gösterildi")
        except Exception as e:
            print(f"Görüntü gösterilirken hata oluştu: {str(e)}")
            messagebox.showerror("Hata", f"Görüntü gösterilirken hata oluştu:\n{str(e)}")

    def to_gray(self):
        if self.image is not None:
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            self.image = gray_3ch
            self.show_image(self.image)
        else:
            messagebox.showwarning("Uyarı", "Önce bir görsel yükleyin.")

    def save_image(self):
        if self.image is not None:
            file_path = filedialog.asksaveasfilename(defaultextension=".png")
            if file_path:
                cv2.imwrite(file_path, self.image)
        else:
            messagebox.showwarning("Uyarı", "Kaydedilecek görsel yok.")

    def show_rgb_channels(self):
        if self.image is not None:
            b, g, r = cv2.split(self.image)
            zeros = np.zeros_like(b)
            # Her kanalı renkli olarak göster
            red_img = cv2.merge([zeros, zeros, r])
            green_img = cv2.merge([zeros, g, zeros])
            blue_img = cv2.merge([b, zeros, zeros])
            print("Mavi kanal min:", b.min(), "max:", b.max())  # Mavi kanal değerlerini yazdır
            cv2.imshow("Red Channel", red_img)
            cv2.imshow("Green Channel", green_img)
            cv2.imshow("Blue Channel", blue_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            messagebox.showwarning("Uyarı", "Önce bir görsel yükleyin.")

    def negative_image(self):
        if self.image is not None:
            negative = 255 - self.image
            self.image = negative
            self.show_image(self.image)
        else:
            messagebox.showwarning("Uyarı", "Önce bir görsel yükleyin.")

    def _on_slider_change(self, _=None):
        """Slider değerleri değiştiğinde çağrılır"""
        if self.original_image is not None:  # Orijinal görüntü varsa
            try:
                img = self.original_image.copy().astype(np.float32)
                brightness = self.brightness_slider.get()
                contrast = self.contrast_slider.get()
                # Parlaklık uygula
                img = img + brightness * 2.55
                # Kontrast uygula
                factor = (259 * (contrast + 255)) / (255 * (259 - contrast)) if contrast != 0 else 1
                img = factor * (img - 128) + 128
                img = np.clip(img, 0, 255).astype(np.uint8)
                self.image = img
                self.show_image(self.image)
            except Exception as e:
                print(f"Parlaklık/kontrast uygulanırken hata oluştu: {str(e)}")

    def show_histogram(self):
        if self.image is not None:
            import matplotlib.pyplot as plt
            img = self.image
            color = ('b', 'g', 'r')
            plt.figure("Histogram")
            for i, col in enumerate(color):
                histr = cv2.calcHist([img], [i], None, [256], [0, 256])
                plt.plot(histr, color=col)
                plt.xlim([0, 256])
            plt.title("RGB Histogramı")
            plt.xlabel("Piksel Değeri")
            plt.ylabel("Piksel Sayısı")
            plt.show()
        else:
            messagebox.showwarning("Uyarı", "Önce bir görsel yükleyin.")

    def equalize_histogram(self):
        if self.image is not None:
            img = self.image
            # Eğer gri ise doğrudan eşitle, renkli ise YCrCb'de sadece Y kanalını eşitle
            if len(img.shape) == 2 or img.shape[2] == 1:
                eq = cv2.equalizeHist(img)
                eq_3ch = cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)
                self.image = eq_3ch
            else:
                img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
                img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
                eq = cv2.cvtColor(img_yuv, cv2.COLOR_YCrCb2BGR)
                self.image = eq
            self.show_image(self.image)
        else:
            messagebox.showwarning("Uyarı", "Önce bir görsel yükleyin.")

    def translate_image(self):
        if self.image is not None:
            rows, cols = self.image.shape[:2]
            M = np.float32([[1, 0, 50], [0, 1, 30]])  # 50 px sağa, 30 px aşağı
            dst = cv2.warpAffine(self.image, M, (cols, rows))
            self.image = dst
            self.show_image(self.image)
        else:
            messagebox.showwarning("Uyarı", "Önce bir görsel yükleyin.")

    def mirror_image(self):
        if self.image is not None:
            mirrored = cv2.flip(self.image, 1)  # Yatay ayna
            self.image = mirrored
            self.show_image(self.image)
        else:
            messagebox.showwarning("Uyarı", "Önce bir görsel yükleyin.")

    def shear_image(self):
        if self.image is not None:
            rows, cols = self.image.shape[:2]
            M = np.float32([[1, 0.5, 0], [0, 1, 0]])  # x ekseninde eğme
            nW = cols + int(rows * 0.5)
            dst = cv2.warpAffine(self.image, M, (nW, rows))
            self.image = dst
            self.show_image(self.image)
        else:
            messagebox.showwarning("Uyarı", "Önce bir görsel yükleyin.")

    def scale_image(self, scale):
        if self.image is not None:
            rows, cols = self.image.shape[:2]
            dst = cv2.resize(self.image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
            self.image = dst
            self.show_image(self.image)
        else:
            messagebox.showwarning("Uyarı", "Önce bir görsel yükleyin.")

    def rotate_image(self):
        if self.image is not None:
            rows, cols = self.image.shape[:2]
            M = cv2.getRotationMatrix2D((cols/2, rows/2), 90, 1)  # 90 derece döndür
            dst = cv2.warpAffine(self.image, M, (cols, rows))
            self.image = dst
            self.show_image(self.image)
        else:
            messagebox.showwarning("Uyarı", "Önce bir görsel yükleyin.")

    def crop_image(self):
        if self.image is not None:
            rows, cols = self.image.shape[:2]
            # Ortadan 100x100 px kırpma (gerekirse boyutları ayarlayın)
            start_row = rows // 2 - 50
            start_col = cols // 2 - 50
            end_row = start_row + 100
            end_col = start_col + 100
            cropped = self.image[max(0, start_row):min(rows, end_row), max(0, start_col):min(cols, end_col)]
            self.image = cropped
            self.show_image(self.image)
        else:
            messagebox.showwarning("Uyarı", "Önce bir görsel yükleyin.")        

    def mean_filter(self):
        if self.image is not None:
            filtered = cv2.blur(self.image, (3, 3))
            self.image = filtered
            self.show_image(self.image)
        else:
            messagebox.showwarning("Uyarı", "Önce bir görsel yükleyin.")

    def median_filter(self):
        if self.image is not None:
            filtered = cv2.medianBlur(self.image, 3)
            self.image = filtered
            self.show_image(self.image)
        else:
            messagebox.showwarning("Uyarı", "Önce bir görsel yükleyin.")

    def gauss_filter(self):
        if self.image is not None:
            filtered = cv2.GaussianBlur(self.image, (3, 3), 0)
            self.image = filtered
            self.show_image(self.image)
        else:
            messagebox.showwarning("Uyarı", "Önce bir görsel yükleyin.")

    def conservative_filter(self):
        if self.image is not None:
            # Her kanal için ayrı ayrı konservatif filtre uygula
            b, g, r = cv2.split(self.image)
            b_f = self._conservative_smoothing_gray(b)
            g_f = self._conservative_smoothing_gray(g)
            r_f = self._conservative_smoothing_gray(r)
            filtered_bgr = cv2.merge([b_f, g_f, r_f])
            self.image = filtered_bgr
            self.show_image(self.image)
        else:
            messagebox.showwarning("Uyarı", "Önce bir görsel yükleyin.")

    def crimmins_speckle_filter(self):
        if self.image is not None:
            # Her kanal için uygula
            b, g, r = cv2.split(self.image)
            b_f = self._crimmins_speckle_iter(b, iterations=7)
            g_f = self._crimmins_speckle_iter(g, iterations=7)
            r_f = self._crimmins_speckle_iter(r, iterations=7)
            filtered_bgr = cv2.merge([b_f, g_f, r_f])
            self.image = filtered_bgr
            self.show_image(self.image)
        else:
            messagebox.showwarning("Uyarı", "Önce bir görsel yükleyin.")

    def _crimmins_speckle_iter(self, img, iterations=3):
        out = img.copy()
        for _ in range(iterations):
            # Azaltma
            for i in range(1, img.shape[0]-1):
                for j in range(1, img.shape[1]-1):
                    neighbors = [out[i-1, j], out[i+1, j], out[i, j-1], out[i, j+1]]
                    avg = sum(neighbors) // 4
                    if out[i, j] > avg:
                        out[i, j] -= 1
            # Artırma
            for i in range(1, img.shape[0]-1):
                for j in range(1, img.shape[1]-1):
                    neighbors = [out[i-1, j], out[i+1, j], out[i, j-1], out[i, j+1]]
                    avg = sum(neighbors) // 4
                    if out[i, j] < avg:
                        out[i, j] += 1
        return out

    def show_fourier(self):
        if self.image is not None:
            import matplotlib.pyplot as plt
            img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            f = np.fft.fft2(img)
            fshift = np.fft.fftshift(f)
            magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
            plt.figure("Fourier Spektrum")
            plt.subplot(121), plt.imshow(img, cmap='gray'), plt.title('Gri Görüntü')
            plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray'), plt.title('Spektrum')
            plt.show()
        else:
            messagebox.showwarning("Uyarı", "Önce bir görsel yükleyin.")

    def low_pass_filter(self):
        self._apply_frequency_filter(self._make_circular_mask, radius=30, highpass=False)

    def high_pass_filter(self):
        self._apply_frequency_filter(self._make_circular_mask, radius=30, highpass=True)

    def band_pass_filter(self):
        self._apply_frequency_filter(self._make_band_mask, r_in=20, r_out=60, bandpass=True)

    def band_stop_filter(self):
        self._apply_frequency_filter(self._make_band_mask, r_in=20, r_out=60, bandpass=False)

    def butterworth_lpf(self):
        self._apply_frequency_filter(self._make_butterworth_mask, radius=30, n=2, highpass=False)

    def butterworth_hpf(self):
        if self.image is not None:
            img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            f = np.fft.fft2(img)
            fshift = np.fft.fftshift(f)
            mask = self._make_butterworth_mask(img.shape, radius=30, n=2, highpass=True)
            fshift_filtered = fshift * mask
            img_back = np.fft.ifft2(np.fft.ifftshift(fshift_filtered)).real
            # Normalize et ve offset ekle
            img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
            img_back = np.uint8(img_back)
            img_back = cv2.add(img_back, 20)  # Daha açık gri için offset
            img_back = np.clip(img_back, 0, 255)
            img_back_bgr = cv2.cvtColor(img_back, cv2.COLOR_GRAY2BGR)
            self.image = img_back_bgr
            self.show_image(self.image)
        else:
            messagebox.showwarning("Uyarı", "Önce bir görsel yükleyin.")

    def gaussian_lpf(self):
        if self.image is not None:
            img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            # Fourier dönüşümü
            f_transform = np.fft.fft2(img)
            f_transform_shifted = np.fft.fftshift(f_transform)
            
            # Gaussian LPF uygula
            glpf = self._make_gaussian_mask(img.shape, radius=30, highpass=False)
            filtered_glpf = f_transform_shifted * glpf
            
            # Ters Fourier dönüşümü
            filtered_image = np.fft.ifft2(np.fft.ifftshift(filtered_glpf)).real
            filtered_image = np.uint8(np.clip(filtered_image, 0, 255))
            
            # Görüntüyü BGR'ye çevir ve göster
            filtered_bgr = cv2.cvtColor(filtered_image, cv2.COLOR_GRAY2BGR)
            self.image = filtered_bgr
            self.show_image(self.image)
        else:
            messagebox.showwarning("Uyarı", "Önce bir görsel yükleyin.")

    def gaussian_hpf(self):
        if self.image is not None:
            img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            # Fourier dönüşümü
            f_transform = np.fft.fft2(img)
            f_transform_shifted = np.fft.fftshift(f_transform)
            
            # Gaussian HPF uygula
            ghpf = self._make_gaussian_mask(img.shape, radius=30, highpass=True)
            filtered_ghpf = f_transform_shifted * ghpf
            
            # Ters Fourier dönüşümü
            filtered_image = np.fft.ifft2(np.fft.ifftshift(filtered_ghpf)).real
            # Normalize et ve offset ekle
            filtered_image = cv2.normalize(filtered_image, None, 0, 255, cv2.NORM_MINMAX)
            filtered_image = np.uint8(filtered_image)
            filtered_image = cv2.add(filtered_image, 20)  # Daha açık gri için offset
            filtered_image = np.clip(filtered_image, 0, 255)
            
            # Görüntüyü BGR'ye çevir ve göster
            filtered_bgr = cv2.cvtColor(filtered_image, cv2.COLOR_GRAY2BGR)
            self.image = filtered_bgr
            self.show_image(self.image)
        else:
            messagebox.showwarning("Uyarı", "Önce bir görsel yükleyin.")

    def homomorphic_filter(self):
        if self.image is not None:
            img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            img_log = np.log1p(np.array(img, dtype="float"))
            M, N = img.shape
            sigma = 30
            gamma1 = 0.5
            gamma2 = 2.0
            (X, Y) = np.meshgrid(np.linspace(-N//2, N//2-1, N), np.linspace(-M//2, M//2-1, M))
            D = np.sqrt(X**2 + Y**2)
            H = (gamma2 - gamma1) * (1 - np.exp(-D**2 / (2 * sigma**2))) + gamma1
            img_fft = np.fft.fftshift(np.fft.fft2(img_log))
            img_fft_filt = H * img_fft
            img_filt = np.fft.ifft2(np.fft.ifftshift(img_fft_filt))
            img_filt = np.exp(np.real(img_filt)) - 1
            img_filt = np.uint8(np.clip(img_filt, 0, 255))
            img_filt_bgr = cv2.cvtColor(img_filt, cv2.COLOR_GRAY2BGR)
            self.image = img_filt_bgr
            self.show_image(self.image)
        else:
            messagebox.showwarning("Uyarı", "Önce bir görsel yükleyin.")

    # Yardımcı fonksiyonlar:
    def _apply_frequency_filter(self, mask_func, **kwargs):
        if self.image is not None:
            img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            f = np.fft.fft2(img)
            fshift = np.fft.fftshift(f)
            mask = mask_func(img.shape, **kwargs)
            fshift_filtered = fshift * mask
            img_back = np.fft.ifft2(np.fft.ifftshift(fshift_filtered))
            img_back = np.abs(img_back)
            img_back = np.uint8(np.clip(img_back, 0, 255))
            img_back_bgr = cv2.cvtColor(img_back, cv2.COLOR_GRAY2BGR)
            self.image = img_back_bgr
            self.show_image(self.image)
        else:
            messagebox.showwarning("Uyarı", "Önce bir görsel yükleyin.")

    def _make_circular_mask(self, shape, radius=30, highpass=False):
        rows, cols = shape
        crow, ccol = rows // 2, cols // 2
        Y, X = np.ogrid[:rows, :cols]
        dist = np.sqrt((X - ccol)**2 + (Y - crow)**2)
        if highpass:
            mask = (dist > radius).astype(np.float32)
        else:
            mask = (dist <= radius).astype(np.float32)
        return mask

    def _make_band_mask(self, shape, r_in=20, r_out=60, bandpass=True):
        rows, cols = shape
        crow, ccol = rows // 2, cols // 2
        Y, X = np.ogrid[:rows, :cols]
        dist = np.sqrt((X - ccol)**2 + (Y - crow)**2)
        if bandpass:
            mask = ((dist >= r_in) & (dist <= r_out)).astype(np.float32)
        else:
            mask = ~((dist >= r_in) & (dist <= r_out))
            mask = mask.astype(np.float32)
        return mask

    def _make_butterworth_mask(self, shape, radius=30, n=2, highpass=False):
        rows, cols = shape
        crow, ccol = rows // 2, cols // 2
        Y, X = np.ogrid[:rows, :cols]
        dist = np.sqrt((X - ccol)**2 + (Y - crow)**2)
        if highpass:
            mask = 1 / (1 + (radius / (dist + 1e-5))**(2*n))
        else:
            mask = 1 / (1 + (dist / (radius + 1e-5))**(2*n))
        return mask.astype(np.float32)

    def _make_gaussian_mask(self, shape, radius=30, highpass=False):
        rows, cols = shape
        crow, ccol = rows // 2, cols // 2
        Y, X = np.ogrid[:rows, :cols]
        dist = np.sqrt((X - ccol)**2 + (Y - crow)**2)
        if highpass:
            mask = 1 - np.exp(-(dist**2) / (2 * (radius**2)))
        else:
            mask = np.exp(-(dist**2) / (2 * (radius**2)))
        return mask.astype(np.float32)

    def sobel_edge(self):
        if self.image is not None:
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            sobel = cv2.magnitude(sobelx, sobely)
            # Daha açık gri için normalize işlemini min=50, max=255 olarak ayarla
            sobel = cv2.normalize(sobel, None, 50, 255, cv2.NORM_MINMAX)
            sobel = np.uint8(sobel)
            sobel_bgr = cv2.cvtColor(sobel, cv2.COLOR_GRAY2BGR)
            self.image = sobel_bgr
            self.show_image(self.image)
        else:
            messagebox.showwarning("Uyarı", "Önce bir görsel yükleyin.")

    def prewitt_x_edge(self):
        if self.image is not None:
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            kernelx = np.array([[1,0,-1],[1,0,-1],[1,0,-1]])
            prewittx = cv2.filter2D(gray, -1, kernelx)
            prewittx_bgr = cv2.cvtColor(prewittx, cv2.COLOR_GRAY2BGR)
            self.image = prewittx_bgr
            self.show_image(self.image)
        else:
            messagebox.showwarning("Uyarı", "Önce bir görsel yükleyin.")

    def prewitt_y_edge(self):
        if self.image is not None:
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            kernely = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
            prewitty = cv2.filter2D(gray, -1, kernely)
            prewitty_bgr = cv2.cvtColor(prewitty, cv2.COLOR_GRAY2BGR)
            self.image = prewitty_bgr
            self.show_image(self.image)
        else:
            messagebox.showwarning("Uyarı", "Önce bir görsel yükleyin.")

    def prewitt_all_edge(self):
        if self.image is not None:
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            kernelx = np.array([[1,0,-1],[1,0,-1],[1,0,-1]])
            kernely = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
            prewittx = cv2.filter2D(gray, -1, kernelx)
            prewitty = cv2.filter2D(gray, -1, kernely)
            prewitt = cv2.magnitude(prewittx.astype(np.float32), prewitty.astype(np.float32))
            prewitt = np.uint8(np.clip(prewitt, 0, 255))
            prewitt_bgr = cv2.cvtColor(prewitt, cv2.COLOR_GRAY2BGR)
            self.image = prewitt_bgr
            self.show_image(self.image)
        else:
            messagebox.showwarning("Uyarı", "Önce bir görsel yükleyin.")

    def roberts_edge(self):
        if self.image is not None:
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            kernelx = np.array([[1, 0], [0, -1]])
            kernely = np.array([[0, 1], [-1, 0]])
            robertsx = cv2.filter2D(gray, -1, kernelx)
            robertsy = cv2.filter2D(gray, -1, kernely)
            roberts = cv2.magnitude(robertsx.astype(np.float32), robertsy.astype(np.float32))
            roberts = np.uint8(np.clip(roberts, 0, 255))
            roberts_bgr = cv2.cvtColor(roberts, cv2.COLOR_GRAY2BGR)
            self.image = roberts_bgr
            self.show_image(self.image)
        else:
            messagebox.showwarning("Uyarı", "Önce bir görsel yükleyin.")

    def compass_edge(self):
        if self.image is not None:
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            # Kirsch kernel seti
            kernels = [
                np.array([[5, 5, 5], [-3, 0, -3], [-3, -3, -3]]),
                np.array([[-3, 5, 5], [-3, 0, 5], [-3, -3, -3]]),
                np.array([[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]]),
                np.array([[-3, -3, -3], [-3, 0, 5], [-3, 5, 5]]),
                np.array([[-3, -3, -3], [-3, 0, -3], [5, 5, 5]]),
                np.array([[-3, -3, -3], [5, 0, -3], [5, 5, -3]]),
                np.array([[5, -3, -3], [5, 0, -3], [5, -3, -3]]),
                np.array([[5, 5, -3], [5, 0, -3], [-3, -3, -3]])
            ]
            max_response = np.zeros_like(gray, dtype=np.float32)
            for k in kernels:
                response = cv2.filter2D(gray, cv2.CV_32F, k)
                max_response = np.maximum(max_response, response)
            compass = np.uint8(np.clip(max_response, 0, 255))
            compass_bgr = cv2.cvtColor(compass, cv2.COLOR_GRAY2BGR)
            self.image = compass_bgr
            self.show_image(self.image)
        else:
            messagebox.showwarning("Uyarı", "Önce bir görsel yükleyin.")

    def canny_edge(self):
        if self.image is not None:
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            self.image = edges_bgr
            self.show_image(self.image)
        else:
            messagebox.showwarning("Uyarı", "Önce bir görsel yükleyin.")

    def laplace_edge(self):
        if self.image is not None:
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            laplacian = np.uint8(np.clip(np.abs(laplacian), 0, 255))
            laplacian_bgr = cv2.cvtColor(laplacian, cv2.COLOR_GRAY2BGR)
            self.image = laplacian_bgr
            self.show_image(self.image)
        else:
            messagebox.showwarning("Uyarı", "Önce bir görsel yükleyin.")

    def gabor_filter(self):
        if self.image is not None:
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            kernel = cv2.getGaborKernel((21, 21), 8.0, np.pi/4, 10.0, 0.5, 0, ktype=cv2.CV_32F)
            filtered = cv2.filter2D(gray, cv2.CV_8UC3, kernel)
            filtered_bgr = cv2.cvtColor(filtered, cv2.COLOR_GRAY2BGR)
            self.image = filtered_bgr
            self.show_image(self.image)
        else:
            messagebox.showwarning("Uyarı", "Önce bir görsel yükleyin.")

    def hough_lines(self):
        if self.image is not None:
            import matplotlib.pyplot as plt
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            lines = cv2.HoughLines(edges, 1, np.pi/180, 100)
            img_lines = self.image.copy()
            if lines is not None:
                for rho, theta in lines[:,0]:
                    a = np.cos(theta)
                    b = np.sin(theta)
                    x0 = a * rho
                    y0 = b * rho
                    x1 = int(x0 + 1000 * (-b))
                    y1 = int(y0 + 1000 * (a))
                    x2 = int(x0 - 1000 * (-b))
                    y2 = int(y0 - 1000 * (a))
                    cv2.line(img_lines, (x1, y1), (x2, y2), (0,0,255), 2)
            self.image = img_lines
            self.show_image(self.image)
        else:
            messagebox.showwarning("Uyarı", "Önce bir görsel yükleyin.")

    def kmeans_segmentation(self):
        if self.image is not None:
            Z = self.image.reshape((-1,3))
            Z = np.float32(Z)
            K = 3
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
            center = np.uint8(center)
            res = center[label.flatten()]
            res2 = res.reshape((self.image.shape))
            self.image = res2
            self.show_image(self.image)
        else:
            messagebox.showwarning("Uyarı", "Önce bir görsel yükleyin.")

    def erode_image(self):
        if self.image is not None:
            kernel = np.ones((3,3), np.uint8)
            eroded = cv2.erode(self.image, kernel, iterations=1)
            self.image = eroded
            self.show_image(self.image)
        else:
            messagebox.showwarning("Uyarı", "Önce bir görsel yükleyin.")

    def dilate_image(self):
        if self.image is not None:
            kernel = np.ones((3,3), np.uint8)
            dilated = cv2.dilate(self.image, kernel, iterations=1)
            self.image = dilated
            self.show_image(self.image)
        else:
            messagebox.showwarning("Uyarı", "Önce bir görsel yükleyin.")

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessorApp(root)
    root.mainloop()