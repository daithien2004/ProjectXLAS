import numpy as np
import cv2
L = 256
def Spectrum(imgin):
    M, N = imgin.shape
    P = cv2.getOptimalDFTSize(M)
    Q = cv2.getOptimalDFTSize(N)
    
    # Bước 1 và 2: 
    # Tạo ảnh mới có kích thước PxQ
    # và thêm số 0 và phần mở rộng
    fp = np.zeros((P,Q), np.float32)
    fp[:M,:N] = imgin
    fp = fp/(L-1)

    # Bước 3:
    # Nhân (-1)^(x+y) để dời vào tâm ảnh
    for x in range(0, M):
        for y in range(0, N):
            if (x+y) % 2 == 1:
                fp[x,y] = -fp[x,y]

    # Bước 4:
    # Tính DFT    
    F = cv2.dft(fp, flags = cv2.DFT_COMPLEX_OUTPUT)

    # Tính spectrum
    S = np.sqrt(F[:,:,0]**2 + F[:,:,1]**2)
    S = np.clip(S, 0, L-1)
    imgout = S.astype(np.uint8)
    return imgout

# def Spectrum(imgin):
#     if len(imgin.shape) > 2 or imgin.dtype != np.uint8:
#         raise ValueError("Đầu vào phải là ảnh xám (np.uint8)")
    
#     # Bước 1: Chuyển sang float32
#     f = imgin.astype(np.float32)
    
#     # Bước 2: Tính FFT 2D
#     F = np.fft.fft2(f)
#     F_shifted = np.fft.fftshift(F)

#     # Bước 3: Lấy độ lớn phổ (magnitude spectrum)
#     magnitude = np.abs(F_shifted)

#     # Bước 4: Dùng log để nén giá trị (có cộng epsilon để tránh log(0))
#     epsilon = 1e-8
#     log_magnitude = np.log(1 + magnitude + epsilon)

#     # Bước 5: Chuẩn hóa về khoảng 0-255
#     log_magnitude_normalized = (log_magnitude - log_magnitude.min()) / (log_magnitude.max() - log_magnitude.min())
#     spectrum_img = (log_magnitude_normalized * 255).astype(np.uint8)
    
#     return spectrum_img


def CreateMoireFilter(M, N):
    H = np.ones((M,N), np.complex64)
    H.imag = 0.0

    u1, v1 = 44, 55
    u2, v2 = 85, 55
    u3, v3 = 41, 111
    u4, v4 = 81, 111

    u5, v5 = M-44, M-55
    u6, v6 = M-85, M-55
    u7, v7 = M-41, M-111
    u8, v8 = M-81, M-111

    D0 = 10
    for u in range(0,M):
        for v in range(0,N):
            # u1, v1
            Duv = np.sqrt((1.0*u-u1)**2 + (1.0*v-v1)**2)
            if Duv <= D0:
                H.real[u,v] = 0.0

            # u2, v2
            Duv = np.sqrt((1.0*u-u2)**2 + (1.0*v-v2)**2)
            if Duv <= D0:
                H.real[u,v] = 0.0

            # u3, v3
            Duv = np.sqrt((1.0*u-u3)**2 + (1.0*v-v3)**2)
            if Duv <= D0:
                H.real[u,v] = 0.0

            # u4, v4
            Duv = np.sqrt((1.0*u-u4)**2 + (1.0*v-v4)**2)
            if Duv <= D0:
                H.real[u,v] = 0.0

            # u5, v5
            Duv = np.sqrt((1.0*u-u5)**2 + (1.0*v-v5)**2)
            if Duv <= D0:
                H.real[u,v] = 0.0
            
            Duv = np.sqrt((1.0*u-u6)**2 + (1.0*v-v6)**2)
            if Duv <= D0:
                H.real[u,v] = 0.0

            Duv = np.sqrt((1.0*u-u7)**2 + (1.0*v-v7)**2)
            if Duv <= D0:
                H.real[u,v] = 0.0

            Duv = np.sqrt((1.0*u-u8)**2 + (1.0*v-v8)**2)
            if Duv <= D0:
                H.real[u,v] = 0.0
    return H


# def DrawInferenceFilter(imgin):
    M, N = imgin.shape
    # Bước 1
    P = cv2.getOptimalDFTSize(M)
    Q = cv2.getOptimalDFTSize(N)
    H = CreateInferenceFilter(P, Q)
    HR = H[:,:,0]
    HR = HR*(L-1)
    imgout = HR.astype(np.uint8)
    return imgout
def DrawInferenceFilter(imgin):
    if len(imgin.shape) > 2:
        raise ValueError("Ảnh đầu vào phải là ảnh grayscale (2D), không phải ảnh màu (3D)")

    M, N = imgin.shape
    H = CreateInferenceFilter(M, N)  # H là mảng 2D kiểu complex64, kích thước (M, N)

    # Dịch chuyển phổ tần số về tâm
    H_shifted = np.fft.fftshift(H)

    # Tính độ lớn của H (magnitude spectrum)
    S = np.sqrt(H_shifted.real**2 + H_shifted.imag**2)
    
    # Chuẩn hóa tuyến tính về dải [0, 255]
    S_min, S_max = S.min(), S.max()
    if S_max > S_min:  # Tránh chia cho 0
        S = (S - S_min) / (S_max - S_min) * 255
    else:
        S = S * 255  # Nếu S_max == S_min, chỉ nhân với 255

    # Chuyển thành ảnh grayscale
    imgout = S.astype(np.uint8)
    return imgout
def CreateInferenceFilter(M,N):
    H = np.ones((M,N), np.complex64)
    H.imag = 0.0
    D0 = 7
    D1 = 7
    for u in range(0,M):
        for v in range(0,N):
            if u not in range(M//2-D0,M//2+D0+1):
                if abs(v-N//2) <= D1:
                    H.real[u,v] = 0.0
    return H


def RemoveMoire(imgin):
    M,N = imgin.shape
    H = CreateMoireFilter(M,N)
    imgout = FrequencyFiltering(imgin, H)
    return imgout

def RemoveInterference(imgin):
    M,N = imgin.shape
    H = CreateInferenceFilter(M,N)
    imgout = FrequencyFiltering(imgin, H)
    return imgout

# def RemoveMoireSimple(imgin):
    M, N = imgin.shape
    # Bước 1
    P = cv2.getOptimalDFTSize(M)
    Q = cv2.getOptimalDFTSize(N)
    fp = np.zeros((P, Q), np.float32)
    # Bước 2
    fp[:M,:N] = 1.0*imgin

    # Bước 3
    for x in range(0,M):
        for y in range(0,N):
            if(x+y) % 2 == 1:
                fp[x,y] = -fp[x,y]
    # Bước 4
    F = cv2.dft(fp, flags=cv2.DFT_COMPLEX_OUTPUT)
    # Bước 5: Tạo bộ lọc H
    H = CreateMoireFilter(P, Q)
    # Bước 6: G = F*H
    G = cv2.mulSpectrums(F, H, flags=cv2.DFT_ROWS)

    # Bước 7: IDFT
    g = cv2.idft(G,flags=cv2.DFT_SCALE)
    # Bước 8: 
    gR = g[:M,:N,0]
    for x in range(0,M):
        for y in range(0,N):
            if(x+y) % 2 == 1:
                gR[x,y] = -gR[x,y]
    gR = np.clip(gR,0,L-1)
    imgout = gR.astype(np.uint8)
    return imgout

def RemoveMoireSimple(imgin):
    M, N = imgin.shape
    P = cv2.getOptimalDFTSize(M)
    Q = cv2.getOptimalDFTSize(N)
    fp = np.zeros((P, Q), np.float32)
    fp[:M,:N] = 1.0 * imgin

    for x in range(0, M):
        for y in range(0, N):
            if (x + y) % 2 == 1:
                fp[x, y] = -fp[x, y]

    # Bước 4: Biến đổi DFT
    F = cv2.dft(fp, flags=cv2.DFT_COMPLEX_OUTPUT)
    
    # Dịch chuyển phổ tần số về tâm
    F_shifted = np.zeros_like(F)
    F_shifted[:, :, 0] = np.fft.fftshift(F[:, :, 0])
    F_shifted[:, :, 1] = np.fft.fftshift(F[:, :, 1])

    # Tạo bộ lọc H
    H = CreateMoireFilter(P, Q)
    H_real = H.real.astype(np.float32)
    H_imag = H.imag.astype(np.float32)
    H_cv = cv2.merge((H_real, H_imag))
    
    # Nhân phổ tần số với bộ lọc
    G = cv2.mulSpectrums(F_shifted, H_cv, flags=cv2.DFT_ROWS)
    
    # Dịch ngược phổ tần số
    G_shifted = np.zeros_like(G)
    G_shifted[:, :, 0] = np.fft.ifftshift(G[:, :, 0])
    G_shifted[:, :, 1] = np.fft.ifftshift(G[:, :, 1])
    
    # Biến đổi ngược IDFT
    g = cv2.idft(G_shifted, flags=cv2.DFT_SCALE)
    
    gR = g[:M, :N, 0]
    for x in range(0, M):
        for y in range(0, N):
            if (x + y) % 2 == 1:
                gR[x, y] = -gR[x, y]
    gR = np.clip(gR, 0, L-1)
    imgout = gR.astype(np.uint8)
    return imgout
# def RemoveInferenceFilter(imgin):
#     M, N = imgin.shape
#     # Bước 1
#     P = cv2.getOptimalDFTSize(M)
#     Q = cv2.getOptimalDFTSize(N)
#     fp = np.zeros((P, Q), np.float32)
#     # Bước 2
#     fp[:M,:N] = 1.0*imgin

#     # Bước 3
#     for x in range(0,M):
#         for y in range(0,N):
#             if(x+y) % 2 == 1:
#                 fp[x,y] = -fp[x,y]
#     # Bước 4
#     F = cv2.dft(fp, flags=cv2.DFT_COMPLEX_OUTPUT)
#     # Bước 5: Tạo bộ lọc H
#     H = CreateInferenceFilter(P, Q)
#     # Bước 6: G = F*H
#     G = cv2.mulSpectrums(F, H, flags=cv2.DFT_ROWS)

#     # Bước 7: IDFT
#     g = cv2.idft(G,flags=cv2.DFT_SCALE)
#     # Bước 8: 
#     gR = g[:M,:N,0]
#     for x in range(0,M):
#         for y in range(0,N):
#             if(x+y) % 2 == 1:
#                 gR[x,y] = -gR[x,y]
#     gR = np.clip(gR,0,L-1)
#     imgout = gR.astype(np.uint8)
#     return imgout

def FrequencyFiltering(imgin, H):
    # Không cần mở rộng ảnh có kích thước PxQ
    f = imgin.astype(np.float32)

    # Bước 1
    F = np.fft.fft2(f)

    # Bước 2
    F = np.fft.fftshift(F)

    # Bước 3: Nhan F voi H, ta được G
    G = F * H

    # Bước 4: Shift G ra trở lại
    G = np.fft.ifftshift(G)

    # Bước 5: IDFT
    g = np.fft.ifft2(G)
    gR = np.clip(g.real, 0, L-1)
    imgout = gR.astype(np.uint8)
    return imgout

def CreateMotionFilter(M,N):
    H = np.zeros((M,N), np.complex64)
    T = 1.0
    a = 0.1
    b = 0.1
    phi_prev = 0.0
    for u in range(0,M):
        for v in range (0,N):
            phi = np.pi*((u-M//2)*a) + ((v-N//2)*b)

            if abs(phi) < 1.0e-6:
                phi = phi_prev

            RE = T*np.sin(phi)*np.cos(phi)/phi
            IM = -T*np.sin(phi)*np.sin(phi)/phi
            H.real[u,v] = RE
            H.imag[u,v] = IM
            phi_prev = phi
    return H

def CreateDemotionFilter(M,N):
    H = np.zeros((M,N), np.complex64)
    T = 1.0
    a = 0.1
    b = 0.1
    phi_prev = 0.0
    for u in range(0,M):
        for v in range (0,N):
            phi = np.pi*((u-M//2)*a) + ((v-N//2)*b)
            mau_so = np.sin(phi)

            if abs(mau_so) < 1.0e-6:
                phi = phi_prev

            RE = phi/(T*np.sin(phi)*np.cos(phi))
            IM = phi/T
            H.real[u,v] = RE
            H.imag[u,v] = IM
            phi_prev = phi
    return H

def CreateMotion(imgin):
    M,N = imgin.shape
    H = CreateMotionFilter(M,N)
    imgout = FrequencyFiltering(imgin, H)
    return imgout

def CreateDemotion(imgin):
    M,N = imgin.shape
    H = CreateDemotionFilter(M,N)
    imgout = FrequencyFiltering(imgin, H)
    return imgout

def CreateDemotionNoise(imgin):
    M, N = imgin.shape
    H = CreateDemotionFilter(M, N)
    img_demotion = FrequencyFiltering(imgin, H)
    imgout = cv2.medianBlur(img_demotion, 5)
    return imgout