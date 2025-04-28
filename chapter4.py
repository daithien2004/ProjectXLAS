import numpy as np
import cv2
L = 256
def Spectrum(imgin):
    # Không cần mở rộng ảnh có kích thước PxQ
    f = imgin.astype(np.float32)

    # Bước 1
    F = np.fft.fft2(f)

    # Bước 2
    F = np.fft.fftshift(F)

    S = np.sqrt(F.real**2 + F.imag**2)
    S = np.clip(S,0,L-1)
    imgout = S.astype(np.uint8)
    return imgout

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

def DrawInferenceFilter(imgin):
    M, N = imgin.shape
    # Bước 1
    P = cv2.getOptimalDFTSize(M)
    Q = cv2.getOptimalDFTSize(N)
    H = CreateInferenceFilter(P, Q)
    HR = H[:,:,0]
    HR = HR*(L-1)
    imgout = HR.astype(np.uint8)
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

def RemoveMoireSimple(imgin):
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


def RemoveInferenceFilter(imgin):
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
    H = CreateInferenceFilter(P, Q)
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