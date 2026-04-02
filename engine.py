import cv2
import numpy as np
from scipy.stats import norm, entropy as shannon_entropy
from skimage.metrics import structural_similarity as ssim
from skimage.feature import hog, local_binary_pattern
from skimage.morphology import skeletonize
from fpdf import FPDF
import datetime
import os
import tempfile

# --- LÓGICA DE PROCESSAMENTO ---

def preprocess_signature(image_path, target_size=(800, 400)):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Arquivo não encontrado: {image_path}")
    try:
        with open(image_path, "rb") as f:
            img_array = np.frombuffer(f.read(), np.uint8)
        img_color = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    except Exception as e:
        raise ValueError(f"Erro de leitura no arquivo {image_path}: {e}")
    if img_color is None:
        raise ValueError(f"Não foi possível carregar a imagem: {image_path}.")
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    blurred = cv2.bilateralFilter(img_gray, d=9, sigmaColor=75, sigmaSpace=75)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError(f"Nenhum traço de assinatura encontrado em: {image_path}")
    img_red_traced = img_color.copy()
    cv2.drawContours(img_red_traced, contours, -1, (0, 0, 255), 1)
    x_min, y_min = float('inf'), float('inf')
    x_max, y_max = 0, 0
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w * h > 100:
            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, x + w)
            y_max = max(y_max, y + h)
    if x_min == float('inf'):
        x_min, y_min, x_max, y_max = 0, 0, closed.shape[1], closed.shape[0]
    margin = 15
    x_min = max(0, x_min - margin)
    y_min = max(0, y_min - margin)
    x_max = min(closed.shape[1], x_max + margin)
    y_max = min(closed.shape[0], y_max + margin)
    cropped_bin = closed[y_min:y_max, x_min:x_max]
    cropped_orig = img_color[y_min:y_max, x_min:x_max]
    cropped_red = img_red_traced[y_min:y_max, x_min:x_max]
    resized_bin = cv2.resize(cropped_bin, target_size, interpolation=cv2.INTER_AREA)
    resized_orig = cv2.resize(cropped_orig, target_size, interpolation=cv2.INTER_AREA)
    resized_red = cv2.resize(cropped_red, target_size, interpolation=cv2.INTER_AREA)
    return {"bin": resized_bin, "orig": resized_orig, "red": resized_red}

def extract_graphotechnical_features(img_color, img_bin):
    mask = img_bin > 127
    ink_pixels = img_color[mask]
    if len(ink_pixels) == 0:
        median_bgr = np.array([0, 0, 0], dtype=np.uint8)
    else:
        median_bgr = np.median(ink_pixels, axis=0).astype(np.uint8)
    bgr_1x1 = np.uint8([[median_bgr]])
    lab_1x1 = cv2.cvtColor(bgr_1x1, cv2.COLOR_BGR2LAB)
    median_lab = lab_1x1[0][0].astype(np.float32)
    contours, _ = cv2.findContours(img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    total_length = 0
    total_vertices = 0
    hesitation_img = img_color.copy()
    for c in contours:
        length = cv2.arcLength(c, True)
        if length > 30:
            total_length += length
            epsilon = 0.005 * length
            approx = cv2.approxPolyDP(c, epsilon, True)
            total_vertices += len(approx)
            for pt in approx:
                px, py = pt[0]
                cv2.circle(hesitation_img, (px, py), 2, (0, 0, 255), -1)
    tremor_index = (total_vertices / total_length * 100) if total_length > 0 else 0
    dist_transform = cv2.distanceTransform(img_bin, cv2.DIST_L2, 5)
    mean_thickness = np.mean(dist_transform[mask]) * 2 if np.sum(mask) > 0 else 0
    if contours:
        all_pts = np.vstack(contours)
        x_g, y_g, w_g, h_g = cv2.boundingRect(all_pts)
        aspect_ratio = h_g / float(w_g) if w_g > 0 else 0
    else:
        aspect_ratio = 0
    bottom_points = []
    angles = []
    rhythm_img = img_color.copy()
    for c in contours:
        if cv2.contourArea(c) > 30:
            x, y, w, h = cv2.boundingRect(c)
            bottom_points.append([x + w/2.0, y + h])
            h_mm = h * 0.0846
            cv2.rectangle(rhythm_img, (x, y), (x+w, y+h), (255, 100, 0), 1)
            cv2.putText(rhythm_img, f"H:{h_mm:.1f}mm", (x, max(10, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 50, 0), 1)
            if len(c) >= 5:
                try:
                    (cx, cy), (MA, ma), angle = cv2.fitEllipse(c)
                    angles.append(angle)
                except Exception:
                    pass
    if len(bottom_points) > 1:
        pts = np.array(bottom_points, dtype=np.float32)
        vx, vy, cx, cy = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
        baseline_slope = (vy[0] / vx[0]) if vx[0] != 0 else 0.0
        baseline_angle = np.degrees(np.arctan(baseline_slope))
    else:
        baseline_angle = 0.0
    axial_angle = np.median(angles) if len(angles) > 0 else 0.0
    total_area = np.sum(mask)
    connectivity_score = (len(contours) / (float(total_area) + 1e-5)) * 10000
    moments = cv2.moments(img_bin)
    hu_moments = cv2.HuMoments(moments).flatten()
    for i in range(7):
        if hu_moments[i] != 0:
            hu_moments[i] = -1 * np.copysign(1.0, hu_moments[i]) * np.log10(abs(hu_moments[i]))
    skel = skeletonize(img_bin > 127)
    skel_uint8 = skel.astype(np.uint8)
    kernel_nb = np.array([[1,1,1],[1,0,1],[1,1,1]], dtype=np.uint8)
    neighbors = cv2.filter2D(skel_uint8, -1, kernel_nb, borderType=cv2.BORDER_CONSTANT)
    endpoints_mask = (skel_uint8 == 1) & (neighbors == 1)
    junctions_mask = (skel_uint8 == 1) & (neighbors > 2)
    endpoints_count = np.sum(endpoints_mask)
    junctions_count = np.sum(junctions_mask)
    skel_img = img_color.copy()
    skel_img = cv2.addWeighted(skel_img, 0.3, np.zeros(skel_img.shape, skel_img.dtype), 0, 0)
    skel_img[skel] = [0, 255, 0]
    ep_y, ep_x = np.where(endpoints_mask)
    for y, x in zip(ep_y, ep_x):
        cv2.circle(skel_img, (x, y), 3, (0, 0, 255), -1)
    jp_y, jp_x = np.where(junctions_mask)
    for y, x in zip(jp_y, jp_x):
        cv2.circle(skel_img, (x, y), 3, (255, 0, 0), -1)
    gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    radius = 1
    n_points = 8 * radius
    lbp = local_binary_pattern(gray, n_points, radius, method="uniform")
    lbp_img = cv2.normalize(lbp, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    lbp_img = cv2.applyColorMap(lbp_img, cv2.COLORMAP_JET)
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    lbp_hist = lbp_hist.astype("float")
    lbp_hist /= (lbp_hist.sum() + 1e-6)
    f_transform = np.fft.fft2(gray)
    f_shift = np.fft.fftshift(f_transform)
    magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1e-6)
    fft_img = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    fft_img = cv2.applyColorMap(fft_img, cv2.COLORMAP_MAGMA)
    h_gray, w_gray = gray.shape
    cy_f, cx_f = h_gray // 2, w_gray // 2
    y_idx, x_idx = np.ogrid[:h_gray, :w_gray]
    mask_area = (x_idx - cx_f)**2 + (y_idx - cy_f)**2 <= 30**2
    low_energy = np.sum(np.abs(f_shift[mask_area]))
    total_energy = np.sum(np.abs(f_shift))
    high_energy = total_energy - low_energy
    fft_energy_ratio = high_energy / (low_energy + 1e-6)
    fourier_desc = np.zeros(32, dtype=np.float64)
    main_contour = None
    if contours:
        main_contour = max(contours, key=cv2.contourArea)
    if main_contour is not None and len(main_contour) >= 8:
        contour_complex = main_contour[:, 0, 0] + 1j * main_contour[:, 0, 1]
        fourier_coeffs = np.fft.fft(contour_complex)
        if abs(fourier_coeffs[1]) > 1e-6:
            fourier_coeffs /= abs(fourier_coeffs[1])
        n_desc = min(32, len(fourier_coeffs) // 2)
        fourier_desc[:n_desc] = np.abs(fourier_coeffs[1:n_desc+1])
    gray_flat = gray.ravel()
    hist_gray, _ = np.histogram(gray_flat, bins=256, range=(0, 256))
    hist_gray = hist_gray.astype(float)
    hist_gray /= (hist_gray.sum() + 1e-6)
    pixel_entropy = float(shannon_entropy(hist_gray + 1e-12))
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sobelx**2 + sobely**2).ravel()
    hist_grad, _ = np.histogram(grad_mag, bins=64)
    hist_grad = hist_grad.astype(float)
    hist_grad /= (hist_grad.sum() + 1e-6)
    grad_entropy = float(shannon_entropy(hist_grad + 1e-12))
    return {
        "color_bgr": median_bgr, "color_lab": median_lab,
        "tremor_index": tremor_index, "thickness": mean_thickness,
        "aspect_ratio": aspect_ratio, "baseline_angle": baseline_angle,
        "axial_angle": axial_angle, "connectivity": connectivity_score,
        "rhythm_img": rhythm_img, "hesitation_img": hesitation_img,
        "hu_moments": hu_moments, "endpoints": int(endpoints_count),
        "junctions": int(junctions_count), "skel_img": skel_img,
        "fft_img": fft_img, "fft_ratio": fft_energy_ratio,
        "lbp_img": lbp_img, "lbp_hist": lbp_hist,
        "fourier_desc": fourier_desc, "pixel_entropy": pixel_entropy,
        "grad_entropy": grad_entropy
    }

def align_images(img_target, img_source):
    warp_mode = cv2.MOTION_TRANSLATION
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    number_of_iterations = 50
    termination_eps = 1e-3
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)
    img1_blur = cv2.GaussianBlur(img_target, (5, 5), 0)
    img2_blur = cv2.GaussianBlur(img_source, (5, 5), 0)
    try:
        _, warp_matrix = cv2.findTransformECC(img1_blur, img2_blur, warp_matrix, warp_mode, criteria, None, 1)
        aligned = cv2.warpAffine(img_source, warp_matrix, (img_target.shape[1], img_target.shape[0]),
                                 flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        return aligned
    except Exception:
        return img_source

def get_grid_density(image, grid=(4, 4)):
    h, w = image.shape
    cell_h = h // grid[0]
    cell_w = w // grid[1]
    densities = []
    for i in range(grid[0]):
        for j in range(grid[1]):
            cell = image[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
            density = np.sum(cell > 0) / (cell.size + 1e-6)
            densities.append(density)
    return np.array(densities)

def extract_features(image):
    hog_feats = hog(image, orientations=9, pixels_per_cell=(16, 16),
                    cells_per_block=(2, 2), block_norm='L2-Hys', visualize=False)
    orb = cv2.ORB_create(nfeatures=500)
    kp, des = orb.detectAndCompute(image, None)
    grid = get_grid_density(image)
    return {"hog": hog_feats, "kp": kp, "des": des, "grid": grid}

def compare_signatures(img1, img2, features1, features2):
    img2_aligned = align_images(img1, img2)
    s_score, _ = ssim(img1, img2_aligned, full=True)
    s_score = max(0, s_score)
    des1 = features1["des"]
    des2 = features2["des"]
    o_score = 0
    if des1 is not None and des2 is not None and len(des1) > 1 and len(des2) > 1:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches = bf.knnMatch(des1, des2, k=2)
        good_matches = []
        for match_tup in matches:
            if len(match_tup) == 2:
                m, n = match_tup
                if m.distance < 0.8 * n.distance:
                    good_matches.append(m)
            elif len(match_tup) == 1:
                good_matches.append(match_tup[0])
        max_possible = max(min(len(des1), len(des2)), 1)
        o_score = min(len(good_matches) / max_possible * 4.0, 1.0)
    hog1 = features1["hog"]
    hog2 = features2["hog"]
    dot_product = np.dot(hog1, hog2)
    norm_a = np.linalg.norm(hog1)
    norm_b = np.linalg.norm(hog2)
    h_score = 0 if (norm_a == 0 or norm_b == 0) else max(0, dot_product / (norm_a * norm_b))
    grid1 = features1["grid"]
    grid2 = features2["grid"]
    mae = np.mean(np.abs(grid1 - grid2))
    g_score = max(0, 1.0 - (mae * 4))
    block_size = 100
    h_b, w_b = img1.shape
    local_ssims = []
    for r in range(0, h_b - block_size, block_size):
        for c in range(0, w_b - block_size, block_size):
            b1 = img1[r:r+block_size, c:c+block_size]
            b2_aligned = img2_aligned[r:r+block_size, c:c+block_size]
            bs, _ = ssim(b1, b2_aligned, full=True)
            if np.sum(b1 > 0) > 50:
                local_ssims.append(max(0, bs))
    local_ssim_score = np.mean(local_ssims) if local_ssims else s_score
    final_score = (0.28 * o_score) + (0.28 * h_score) + (0.16 * s_score) + (0.16 * g_score) + (0.12 * local_ssim_score)
    return final_score

def analyze_signatures(questioned_path, sample_paths, log_callback=print):
    log_callback(f"[INFO] Processando a Assinatura Questionada...")
    try:
        q_prep = preprocess_signature(questioned_path)
        q_img = q_prep["bin"]
        q_features = extract_features(q_img)
        q_morph = extract_graphotechnical_features(q_prep["orig"], q_img)
    except Exception as e:
        return {"error": f"Erro na questionada: {str(e)}"}
    s_imgs = []; s_origs = []; s_reds = []; s_feats = []; s_morphs = []
    log_callback(f"[INFO] Extraindo geometria das amostras autênticas ({len(sample_paths)} encontradas)...")
    for idx, sp in enumerate(sample_paths):
        try:
            prep = preprocess_signature(sp)
            s_bin = prep["bin"]
            feat = extract_features(s_bin)
            morph = extract_graphotechnical_features(prep["orig"], s_bin)
            s_imgs.append(s_bin); s_origs.append(prep["orig"])
            s_reds.append(prep["red"]); s_feats.append(feat); s_morphs.append(morph)
        except Exception as e:
            log_callback(f"[AVISO] Ignorando amostra {idx + 1}: {str(e)}")
    if len(s_imgs) < 2:
        return {"error": "O algoritmo exige no mínimo 2 amostras autênticas válidas."}
    log_callback(f"[PROCESSAMENTO] Calculando scores entre amostras...")
    intra_scores = []
    for i in range(len(s_imgs)):
        for j in range(i + 1, len(s_imgs)):
            score = compare_signatures(s_imgs[i], s_imgs[j], s_feats[i], s_feats[j])
            intra_scores.append(score)
    mean_score = np.mean(intra_scores)
    std_score = np.std(intra_scores)
    if std_score < 0.05:
        std_score = 0.05
    log_callback(f"[COMPARAÇÃO] Comparando questionada com amostras...")
    q_scores = []
    for i in range(len(s_imgs)):
        score = compare_signatures(s_imgs[i], q_img, s_feats[i], q_features)
        q_scores.append(score)
    mean_q_score = np.mean(q_scores)
    z_score = (mean_q_score - mean_score) / std_score
    if mean_q_score >= mean_score:
        probability = 99.99
    else:
        probability = np.exp(z_score / 1.5) * 100
    if probability >= 70:
        conclusion = "Autêntica. Consistência geométrica, caligrafia e paradas de caneta coincidem fortemente."
    elif 35 <= probability < 70:
        conclusion = "Inconclusivo. A geometria macro é similar, mas os Keypoints finos apresentam divergência."
    else:
        conclusion = "FALSIDADE. Incompatibilidade geométrica pesada apontando forte fraude."
    advanced_flags = []
    def calc_z(key, threshold, flag_msg):
        vals = [m[key] for m in s_morphs]
        if isinstance(vals[0], np.ndarray):
            mean_v = np.mean(vals, axis=0)
            q_val = q_morph[key]
            dist = np.linalg.norm(q_val - mean_v)
            if dist > threshold:
                advanced_flags.append(flag_msg)
        else:
            mean_v = np.mean(vals)
            std_v = np.std(vals)
            if std_v < 0.1: std_v = 0.1
            q_val = q_morph[key]
            z = abs(q_val - mean_v) / std_v
            if z > threshold:
                advanced_flags.append(flag_msg)
    calc_z("color_lab", 45.0, "[!] Distinção Crítica Colorimétrica (Delta-E CIE76): A tinta difere drasticamente do padrão da base.")
    calc_z("tremor_index", 2.5, "[!] Indício de Falsificação Lenta (Tremor): Traço altamente poligonizado e hesitante.")
    calc_z("thickness", 2.5, "[!] Divergência de Pressão Gráfica: A espessura média destoa do padrão autêntico.")
    calc_z("aspect_ratio", 2.5, "[!] Proporcionalidade Gráfica Incompatível: Relação Altura/Largura não bate com o autor.")
    calc_z("baseline_angle", 2.5, "[!] Alinhamento Gráfico Atípico: O caimento da linha de base difere do natural.")
    calc_z("axial_angle", 2.5, "[!] Inclinação Gráfica Divergente: Padrão axial incompatível.")
    calc_z("connectivity", 2.5, "[!] Quebra de Ritmo/Conectividade: Componentes destoa radicalmente da matriz.")
    calc_z("hu_moments", 15.0, "[!] Identidade Física Falsa (Momentos de Hu): Peso estrutural e inércia geométrica falham.")
    calc_z("endpoints", 3.0, "[!] Inconsistência Topológica de Nós (Ataques/Remates): Quebras livres destoa do esqueleto.")
    calc_z("junctions", 3.0, "[!] Divergência de Cruzamentos/Bifurcações: Ilhas de cruzamento diferem da topologia autêntica.")
    calc_z("fft_ratio", 4.0, "[!] Frequência Espectral Incompatível (FFT): Distribuição de tremores físicos severamente alterada.")
    s_lbps = [m["lbp_hist"] for m in s_morphs]
    mean_lbp = np.mean(s_lbps, axis=0)
    lbp_dist = np.linalg.norm(q_morph["lbp_hist"] - mean_lbp)
    if lbp_dist > 0.15:
        advanced_flags.append("[!] Micro-Textura Anômala (LBP): Divergência no atrito ou deposição da tinta no papel.")
    s_fous = np.array([m["fourier_desc"] for m in s_morphs])
    mean_fou = np.mean(s_fous, axis=0)
    fou_dist = np.linalg.norm(q_morph["fourier_desc"] - mean_fou)
    fou_std = np.mean(np.std(s_fous, axis=0)) + 1e-6
    if fou_dist / fou_std > 4.0:
        advanced_flags.append("[!] Estrutura de Contorno Incompatível (Fourier Descriptors): Perfil matemático diverge do modelo autêntico.")
    calc_z("pixel_entropy", 3.0, "[!] Complexidade Gráfica Atípica (Entropia de Pixel): Densidade de informação visual difere do padrão.")
    calc_z("grad_entropy", 3.0, "[!] Complexidade de Gradientes Anômala (Entropia de Gradientes): Distribuição angular difere do padrão neuromotor.")
    summary_metrics = {
        "q_thickness": q_morph["thickness"], "q_axial": q_morph["axial_angle"],
        "q_aspect": q_morph["aspect_ratio"], "q_endpoints": q_morph["endpoints"],
        "q_junctions": q_morph["junctions"], "q_fft": q_morph["fft_ratio"],
        "q_fourier_dist": float(fou_dist), "q_pixel_entropy": q_morph["pixel_entropy"],
        "q_grad_entropy": q_morph["grad_entropy"],
        "s_thickness": np.mean([m["thickness"] for m in s_morphs]),
        "s_axial": np.mean([m["axial_angle"] for m in s_morphs]),
        "s_aspect": np.mean([m["aspect_ratio"] for m in s_morphs]),
        "s_endpoints": np.mean([m["endpoints"] for m in s_morphs]),
        "s_junctions": np.mean([m["junctions"] for m in s_morphs]),
        "s_fft": np.mean([m["fft_ratio"] for m in s_morphs]),
        "s_pixel_entropy": np.mean([m["pixel_entropy"] for m in s_morphs]),
        "s_grad_entropy": np.mean([m["grad_entropy"] for m in s_morphs])
    }
    log_callback("[SUCESSO] Análise Concluída!\n")
    return {
        "mean_sample_similarity": mean_score, "sample_std": std_score,
        "questioned_similarity": mean_q_score, "q_scores": q_scores,
        "z_score": z_score, "probability_percentage": probability,
        "conclusion": conclusion, "advanced_flags": advanced_flags,
        "summary_metrics": summary_metrics,
        "q_rhythm": q_morph["rhythm_img"], "q_hesitation": q_morph["hesitation_img"],
        "q_skel": q_morph["skel_img"], "q_fft": q_morph["fft_img"],
        "q_lbp": q_morph["lbp_img"], "q_img": q_img,
        "q_orig": q_prep["orig"], "q_red": q_prep["red"],
        "s_imgs": s_imgs, "s_origs": s_origs, "s_reds": s_reds,
        "s_skels": [m["skel_img"] for m in s_morphs],
        "s_ffts": [m["fft_img"] for m in s_morphs],
        "s_lbps": [m["lbp_img"] for m in s_morphs]
    }

# --- MÓDULO DE GERAÇÃO DE PDF ---
class PDFReport(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 15)
        self.set_text_color(41, 128, 185)
        self.cell(0, 10, 'Laudo Pericial - Analise Forense de Assinaturas (GrafoAnalyzer PreciSion V2)', 0, 1, 'C')
        self.set_font('Helvetica', 'I', 10)
        self.set_text_color(127, 140, 141)
        self.cell(0, 10, f'Data de Geracao: {datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")}', 0, 1, 'C')
        self.ln(5)
    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(149, 165, 166)
        self.cell(0, 10, f'Pagina {self.page_no()} | GrafoAnalyzer PreciSion V2 - Laudo Forense Confidencial', 0, 0, 'C')

def generate_pdf_report(results_list, save_dir=None, return_bytes=False):
    """
    Gera o laudo PDF.
    - Se return_bytes=True: retorna bytes do PDF (para Streamlit download_button)
    - Se return_bytes=False: salva em save_dir e retorna o caminho do arquivo
    """
    _tmp_dir = None
    if save_dir is None:
        _tmp_dir = tempfile.mkdtemp()
        save_dir = _tmp_dir
    else:
        os.makedirs(save_dir, exist_ok=True)

    temp_files = []
    # Orientacao paisagem A4: largura=297mm, altura=210mm
    pdf = PDFReport(orientation='L', unit='mm', format='A4')

    def enc(s):
        return s.encode('latin-1', 'replace').decode('latin-1')

    def pdf_section_title(title):
        pdf.set_font('Helvetica', 'B', 13)
        pdf.set_text_color(41, 128, 185)
        pdf.cell(0, 10, enc(title), 0, 1)
        pdf.set_draw_color(41, 128, 185)
        pdf.line(15, pdf.get_y(), 282, pdf.get_y())
        pdf.ln(3)

    def pdf_method_block(method_name, equation_lines, explanation):
        pdf.set_font('Helvetica', 'B', 11)
        pdf.set_text_color(44, 62, 80)
        pdf.cell(0, 8, enc(method_name), 0, 1)
        pdf.set_fill_color(236, 240, 241)
        pdf.set_font('Courier', 'B', 10)
        pdf.set_text_color(52, 73, 94)
        for eq in equation_lines:
            pdf.cell(0, 7, enc('   ' + eq), 0, 1, fill=True)
        pdf.ln(2)
        pdf.set_font('Helvetica', '', 9)
        pdf.set_text_color(74, 85, 104)
        pdf.multi_cell(0, 5, enc(explanation))
        pdf.ln(5)

    # === PÁGINA 1: METODOLOGIA ===
    pdf.add_page()
    pdf.set_font('Helvetica', 'B', 14)
    pdf.set_text_color(44, 62, 80)
    pdf.cell(0, 10, '1. Introducao: Metodologia Cientifica e Base Legal', 0, 1)
    pdf.set_text_color(52, 73, 94)
    pdf.set_font('Helvetica', '', 10)
    method_text = (
        "O presente laudo pericial contem um compilado de analises realizadas em sequencia pelo motor GrafoAnalyzer PreciSion V2. "
        "A similaridade base (motor pericial) foi transversalmente dividida entre cinco metodologias algoritmicas distintas:\n\n"
        "A) Algoritmo ORB - 28% do peso final: (Oriented FAST e Rotated BRIEF). Mapeia esquinas, lacas e conectores "
        "chave cruzando coordenadas organicas via distancia de Hamming entre descritores binarios.\n\n"
        "B) Algoritmo HOG - 28% do peso final: (Histograma de Gradientes Orientados). Observa a variancia macro "
        "dos tracos por meio de gradientes orientados em celulas de 16x16 pixels, normalizado por L2-Hys.\n\n"
        "C) Alinhamento ECC e SSIM - 16% do peso final: Alinha geometricamente por translacao (ECC) e compara "
        "estrutura, luminancia e contraste simultaneamente (SSIM - Wang et al., 2004).\n\n"
        "D) SSIM Local por Blocos - 12% do peso final: Detecta falsificacoes parciais avaliando blocos de 100x100px "
        "individualmente, evitando que desvios localizados sejam mascarados pela media global.\n\n"
        "E) Distribuicao em Grade Espacial - 16% do peso final: Divide a assinatura em 16 quadrantes (grade 4x4) "
        "e rastreia o equilibrio gravitacional da tinta por celula.\n\n"
        f"TOTAL DE ANALISES NESTE RELATORIO: {len(results_list)}"
    )
    pdf.multi_cell(0, 6, enc(method_text))

    # === PÁGINAS DE FUNDAMENTOS MATEMÁTICOS ===
    pdf.add_page()
    pdf_section_title("2. Fundamentos Matematicos dos Modulos de Extracao")
    pdf.set_font('Helvetica', 'I', 9); pdf.set_text_color(100, 100, 100)
    pdf.multi_cell(0, 5, enc("Esta secao documenta as equacoes e fundamentos matematicos de cada modulo de extracao de caracteristicas."))
    pdf.ln(5)
    pdf_method_block("2.1  ORB - Deteccao FAST e Descritor BRIEF Rotacionado (Peso: 28%)",
        ["Detector FAST:  Se |I(p) - I(x)| > t para N de 16 vizinhos => Ponto-Chave detectado",
         "Orientacao:     theta = arctan2( m_01, m_10 )  onde m_pq = Sum(x^p * y^q * I(x,y))",
         "Pontuacao ORB: min(|boas_corr|, n_feats) / max(|des1|, |des2|) * 4.0"],
        "O ORB combina o detector de cantos FAST com o descritor binario BRIEF rotacionado pelo angulo calculado "
        "a partir dos momentos de imagem. A busca por correspondencias usa a distancia de Hamming (numero de bits "
        "diferentes entre descritores). O fator 4.0 normaliza a escala para [0, 1].")
    pdf_method_block("2.2  HOG - Histograma de Gradientes Orientados (Peso: 28%)",
        ["Gradiente:  Gx = I*[-1,0,1], Gy = I*[-1,0,1]^T",
         "Magnitude:  |G| = sqrt(Gx^2 + Gy^2),  Orientacao: theta = arctan2(Gy, Gx) mod 180",
         "Pontuacao HOG:  cos(HOG_1, HOG_2) = (HOG_1.HOG_2) / (||HOG_1|| * ||HOG_2||)"],
        "O HOG descreve a distribuicao macro das direcoes dos tracos da assinatura. Cada celula de 16x16px vota "
        "em 9 orientacoes ponderadas pela magnitude do gradiente. Normalizacao L2-Hys elimina variacoes de contraste.")
    pdf_method_block("2.3  Alinhamento ECC e SSIM Local por Blocos (Pesos: 16% + 12%)",
        ["ECC: rho(T,I;W) = (T-media(T))^T * W(I) / (||T-media(T)||_F * ||W(I)-media(W(I))||_F)",
         "SSIM: S(x,y) = (2*ux*uy+C1)(2*sxy+C2) / ((ux^2+uy^2+C1)(sx^2+sy^2+C2))",
         "SSIM Local: media({ ssim_b : soma(bloco > 0) > 50 })"],
        "O ECC alinha geometricamente por translacao antes de qualquer comparacao. O SSIM mede luminancia, "
        "contraste e estrutura simultaneamente (mais proximo da percepcao humana que o MSE). "
        "O SSIM Local em blocos de 100x100px detecta falsificacoes parciais.")
    pdf_method_block("2.4  Distribuicao em Grade Espacial (Peso: 16%)",
        ["Grade 4x4 (16 celulas): densidade_ij = soma(I[i,j]>0) / area_celula",
         "EAM = (1/16)*Soma_ij |densidade_ij_ref - densidade_ij_quest|",
         "Pontuacao em Grade = max(0, 1.0 - EAM*4.0)"],
        "A assinatura e dividida em 16 quadrantes. Para cada celula, calcula-se a fracao de pixels de tinta. "
        "Assinaturas genuinas mantem padrao espacial caracteristico do autor. O fator 4.0 amplifica a penalidade.")

    pdf.add_page()
    pdf_section_title("2. Fundamentos Matematicos (continuacao) - Metricas Grafotecnicas")
    pdf_method_block("2.5  Pressao Grafica via Transformada de Distancia Euclidiana",
        ["dist_transf(p) = min{ d(p,q) : q pertence ao fundo }  (Distancia Euclidiana L2)",
         "Espessura_media = 2 * media( dist_transf(p) : p pertence a tinta )"],
        "A Transformada de Distancia L2 atribui a cada pixel de tinta sua distancia ao pixel de fundo mais proximo. "
        "O valor medio * 2 equivale ao diametro medio do traco, diretamente proporcional a pressao exercida na caneta.")
    pdf_method_block("2.6  Topologia Esqueletica - Terminais e Cruzamentos",
        ["Esqueleto: skel = skeletonize(I_bin)  [Algoritmo de Afinamento Zhang-Suen]",
         "N(p) = conv2d(skel, K)  onde K=[[1,1,1],[1,0,1],[1,1,1]] (vizinhanca 8-conectada)",
         "Terminais (Endpoints): N(p)=1,  Cruzamentos (Junctions): N(p)>2"],
        "Reduz tracos a 1px preservando conectividade topologica. Terminais representam ataques e remates "
        "(onde a caneta inicia ou termina). Cruzamentos representam bifurcacoes e ligacoes de letras cursivas.")
    pdf_method_block("2.7  Espectro de Frequencia FFT e Entropia de Shannon",
        ["F(u,v) = Soma_x Soma_y I(x,y)*exp(-j*2*pi*(ux/M+vy/N))",
         "Razao de Energia: E_ratio = E_alta / E_baixa  (mascara radial r<=30px)",
         "Entropia de Shannon: H = -Soma_i p_i * log(p_i)  (histograma de 256 niveis)"],
        "A FFT 2D decompoe a imagem em frequencias espaciais. Escritas naturais em alta velocidade geram E_ratio "
        "elevado (ricas em altas frequencias). A Entropia de Shannon mede a complexidade grafomotora neural: "
        "escritas naturais possuem entropia caracteristica estavel, enquanto falsificacoes apresentam valores anomalos.")

    # === SECAO 3: MOTOR PROBABILISTICO ===
    pdf.add_page()
    pdf_section_title("3. Motor Probabilistico: Z-Score, Probabilidade e Raciocinio Final")
    pdf_method_block("3.1  Pontuacao Composta de Similaridade (Fusao Ponderada)",
        ["Pontuacao(A, B) = 0.28*ORB + 0.28*HOG + 0.16*SSIM + 0.16*Grade + 0.12*SSIMLoc",
         "Cada componente em [0,1],  Pontuacao_final em [0,1]"],
        "Fusao ponderada de 5 algoritmos heterogeneos. Os pesos foram calibrados para maximizar a separabilidade "
        "entre assinaturas genuinas e falsificadas.")
    pdf_method_block("3.2  Z-Score Geometrico e Mapeamento Probabilistico",
        ["mu_base = media(pontuacoes intra-amostras),  sigma_base = desvio_padrao(pontuacoes)",
         "Z = (mu_Q - mu_base) / sigma_base",
         "P = exp(Z/1.5)*100 se mu_Q < mu_base,  caso contrario P = 99,99%",
         "Autentica >= 70%,  Inconclusiva 35-70%,  Falsidade < 35%"],
        "O Z-Score mede a posicao estatistica da questionada em relacao a distribuicao autentica, em unidades de "
        "desvio padrao. O mesmo raciocinio e usado em controle de qualidade (Seis Sigma) e testes de hipotese.")

    # === LOOP POR ANÁLISE ===
    for i, results_dict in enumerate(results_list):
        q_orig_path = os.path.join(save_dir, f"tmp_q_orig_{i}.png")
        q_rhyt_path = os.path.join(save_dir, f"tmp_q_rhyt_{i}.png")
        q_hes_path  = os.path.join(save_dir, f"tmp_q_hes_{i}.png")
        q_skel_path = os.path.join(save_dir, f"tmp_q_skel_{i}.png")
        q_red_path  = os.path.join(save_dir, f"tmp_q_red_{i}.png")
        q_bin_path  = os.path.join(save_dir, f"tmp_q_bin_{i}.png")
        q_fft_path  = os.path.join(save_dir, f"tmp_q_fft_{i}.png")
        q_lbp_path  = os.path.join(save_dir, f"tmp_q_lbp_{i}.png")
        cv2.imwrite(q_orig_path, results_dict["q_orig"])
        cv2.imwrite(q_rhyt_path, results_dict["q_rhythm"])
        cv2.imwrite(q_hes_path,  results_dict["q_hesitation"])
        cv2.imwrite(q_skel_path, results_dict["q_skel"])
        cv2.imwrite(q_red_path,  results_dict["q_red"])
        cv2.imwrite(q_bin_path,  results_dict["q_img"])
        cv2.imwrite(q_fft_path,  results_dict["q_fft"])
        cv2.imwrite(q_lbp_path,  results_dict["q_lbp"])
        temp_files.extend([q_orig_path, q_rhyt_path, q_hes_path, q_skel_path,
                           q_red_path, q_bin_path, q_fft_path, q_lbp_path])
        s_paths = []
        for idx in range(len(results_dict["s_imgs"])):
            po  = os.path.join(save_dir, f"tmp_s_o_{i}_{idx}.png")
            pr  = os.path.join(save_dir, f"tmp_s_r_{i}_{idx}.png")
            pb  = os.path.join(save_dir, f"tmp_s_b_{i}_{idx}.png")
            psk = os.path.join(save_dir, f"tmp_s_skel_{i}_{idx}.png")
            pf  = os.path.join(save_dir, f"tmp_s_fft_{i}_{idx}.png")
            pl  = os.path.join(save_dir, f"tmp_s_lbp_{i}_{idx}.png")
            cv2.imwrite(po, results_dict["s_origs"][idx])
            cv2.imwrite(pr, results_dict["s_reds"][idx])
            cv2.imwrite(pb, results_dict["s_imgs"][idx])
            temp_files.extend([po, pr, pb])
            s_dict = {"orig": po, "red": pr, "bin": pb}
            s_skels = results_dict.get("s_skels", [])
            s_ffts  = results_dict.get("s_ffts", [])
            s_lbps  = results_dict.get("s_lbps", [])
            if idx < len(s_skels): cv2.imwrite(psk, s_skels[idx]); temp_files.append(psk); s_dict["skel"] = psk
            if idx < len(s_ffts):  cv2.imwrite(pf,  s_ffts[idx]);  temp_files.append(pf);  s_dict["fft"]  = pf
            if idx < len(s_lbps):  cv2.imwrite(pl,  s_lbps[idx]);  temp_files.append(pl);  s_dict["lbp"]  = pl
            s_paths.append(s_dict)

        pdf.add_page()
        pdf.set_font('Helvetica', 'B', 16); pdf.set_text_color(41, 128, 185)
        pdf.cell(0, 12, enc(f'--- ANALISE PERICIAL N. {i+1} ---'), 0, 1, 'C'); pdf.ln(5)

        # Secao A: Questionada
        # Paisagem A4: largura=297mm. Margem 15mm de cada lado => usavel=267mm. x_centro=(297-240)/2=28,5
        pdf.set_font('Helvetica', 'B', 14); pdf.set_text_color(44, 62, 80)
        pdf.cell(0, 10, 'A. Assinatura Questionada (Documento Suspeito)', 0, 1)
        for label, path in [('1. Imagem Original Colhida do Documento:', q_orig_path),
                            ('2. Tracado Biometrico de Mapeamento (Contornos em Vermelho):', q_red_path),
                            ('3. Binarizacao Matematica pelo Motor de Visao Computacional:', q_bin_path),
                            ('4. Ritmo Grafico Estrutural (Espacamento estimado em milimetros):', q_rhyt_path),
                            ('5. Mapeamento de Pontos de Hesitacao (Tremor e Paradas):', q_hes_path),
                            ('6. Analise Topologica do Esqueleto (Terminais em Verm. / Cruzamentos em Azul):', q_skel_path)]:
            pdf.set_font('Helvetica', 'B', 12); pdf.set_text_color(192, 57, 43)
            pdf.cell(0, 10, label, 0, 1, 'C')
            # Paisagem: largura pagina 297, imagem w=240, x=(297-240)/2=28.5
            pdf.image(path, x=28.5, w=240); pdf.ln(5)
        pdf.set_font('Helvetica', 'B', 12); pdf.set_text_color(192, 57, 43)
        pdf.cell(0, 10, '7. Analise Microscopica: Espectro Radial FFT (esq.) e Micro-Textura LBP (dir.):', 0, 1, 'C')
        y_s7 = pdf.get_y()
        # Dois blocos lado a lado com mais espaco em paisagem
        pdf.image(q_fft_path, x=20, y=y_s7, w=125)
        pdf.image(q_lbp_path, x=152, y=y_s7, w=125); pdf.ln(65)

        # Secao B: Amostras
        # Paisagem: img_w=100, sx=30 => 2 imagens lado a lado: 25 + 100 + 30 + 100 = 255 < 282
        pdf.add_page()
        pdf.set_font('Helvetica', 'B', 14); pdf.set_text_color(44, 62, 80)
        pdf.cell(0, 10, 'B. Amostras Autenticas Parametrizadas (Base de Calibracao)', 0, 1)
        pdf.set_font('Helvetica', '', 10); pdf.set_text_color(52, 73, 94)
        pdf.multi_cell(0, 5, enc("Para cada amostra sao exibidas seis camadas de analise: "
            "Original | Tracado Biometrico | Binarizado | Topologia do Esqueleto | Espectro FFT | Micro-Textura LBP"))
        pdf.ln(4)
        for idx, sp in enumerate(s_paths):
            if idx > 0: pdf.add_page()
            pdf.set_font('Helvetica', 'B', 11); pdf.set_text_color(39, 174, 96)
            pdf.cell(0, 8, enc(f'Amostra Autentica {idx+1} de {len(s_paths)}:'), 0, 1)
            img_w = 100; sx = 30
            y_p = pdf.get_y()
            pdf.set_font('Helvetica', 'B', 8); pdf.set_text_color(52, 73, 94)
            pdf.text(25, y_p+2, "Original Colhida"); pdf.text(25+img_w+sx, y_p+2, "Tracado Biometrico (Contornos)")
            pdf.image(sp["orig"], x=25, y=y_p+4, w=img_w)
            pdf.image(sp["red"],  x=25+img_w+sx, y=y_p+4, w=img_w); pdf.ln(55)
            y_p = pdf.get_y()
            pdf.text(25, y_p+2, "Binarizado (Motor CV)"); pdf.text(25+img_w+sx, y_p+2, "Topologia do Esqueleto (Terminais/Cruzamentos)")
            pdf.image(sp["bin"], x=25, y=y_p+4, w=img_w)
            if "skel" in sp: pdf.image(sp["skel"], x=25+img_w+sx, y=y_p+4, w=img_w)
            pdf.ln(55)
            y_p = pdf.get_y()
            pdf.text(25, y_p+2, "Espectro de Frequencia (FFT)"); pdf.text(25+img_w+sx, y_p+2, "Micro-Textura da Tinta (LBP)")
            if "fft" in sp: pdf.image(sp["fft"], x=25, y=y_p+4, w=img_w)
            if "lbp" in sp: pdf.image(sp["lbp"], x=25+img_w+sx, y=y_p+4, w=img_w)
            pdf.ln(55)

        # Secao C: Resultado Quantitativo
        pdf.add_page()
        pdf.set_font('Helvetica', 'B', 14); pdf.set_text_color(44, 62, 80)
        pdf.cell(0, 10, enc(f'C. Laudo Quantitativo Final — Analise N. {i+1}'), 0, 1)
        pdf.set_font('Courier', '', 11); pdf.set_text_color(0, 0, 0)
        metrics = results_dict.get('summary_metrics', {})
        res_text = (
            f"Consistencia Inter-Amostras (Linha de Base):       {results_dict['mean_sample_similarity']:.2%}\n"
            f"Similaridade Media da Questionada:                 {results_dict['questioned_similarity']:.2%}\n"
            f"Desvio Padrao Natural da Base:                     {results_dict['sample_std']:.4f}\n"
            f"Z-Score Geometrico (Posicao Estatistica):          {results_dict['z_score']:.2f}\n"
            f"------------------------------------------------------------------\n"
            f"Mapeamento Probabilistico de Autenticidade:        {results_dict['probability_percentage']:.2f}%\n"
        )
        if metrics:
            res_text += (
                f"------------------------------------------------------------------\n"
                f"[ METRICAS GRAFOTECNICAS CLASSIFICADAS ]\n"
                f"Pressao Grafica (Espessura):  Quest.={metrics['q_thickness']:.2f}px    | Base={metrics['s_thickness']:.2f}px\n"
                f"Inclinacao Axial:             Quest.={metrics['q_axial']:.1f} graus  | Base={metrics['s_axial']:.1f} graus\n"
                f"Proporcionalidade (A/L):      Quest.={metrics['q_aspect']:.3f}        | Base={metrics['s_aspect']:.3f}\n"
                f"Terminais / Cruzamentos:      Quest.=T:{metrics['q_endpoints']:.0f}/C:{metrics['q_junctions']:.0f}       | Base=T:{metrics['s_endpoints']:.0f}/C:{metrics['s_junctions']:.0f}\n"
                f"Energia Espectral (FFT):      Quest.={metrics['q_fft']:.2f}           | Base={metrics['s_fft']:.2f}\n"
                f"Entropia de Pixel (nats):     Quest.={metrics.get('q_pixel_entropy',0):.3f}        | Base={metrics.get('s_pixel_entropy',0):.3f}\n"
                f"Entropia de Gradientes(nats): Quest.={metrics.get('q_grad_entropy',0):.3f}        | Base={metrics.get('s_grad_entropy',0):.3f}\n"
                f"Dist. Descritores Fourier:    Quest.={metrics.get('q_fourier_dist',0):.3f}        (menor = mais similar)\n"
            )
        pdf.multi_cell(0, 6, enc(res_text)); pdf.ln(5)

        pdf.set_font('Helvetica', 'B', 12); pdf.set_text_color(44, 62, 80)
        pdf.cell(0, 8, 'CONCLUSAO MATEMATICA CONJUNTA:', 0, 1)
        prob = results_dict['probability_percentage']
        if prob >= 70:   pdf.set_text_color(39, 174, 96)
        elif prob >= 35: pdf.set_text_color(243, 156, 18)
        else:            pdf.set_text_color(192, 57, 43)
        pdf.set_font('Helvetica', 'B', 14)
        pdf.multi_cell(0, 8, enc(results_dict['conclusion'])); pdf.ln(5)

        flags = results_dict.get('advanced_flags', [])
        if flags:
            pdf.set_font('Helvetica', 'B', 12); pdf.set_text_color(192, 57, 43)
            pdf.cell(0, 8, '-> AVISOS DE DISCREPANCIA EXTRAMÉTRICA (BIODINAMICA / TINTA):', 0, 1)
            pdf.set_font('Helvetica', 'I', 10)
            for f in flags:
                pdf.multi_cell(0, 6, enc(f)); pdf.ln(2)

    # === RESUMO EXECUTIVO ===
    # Paisagem: colunas totais = 20+115+40+47+45 = 267 => cabe em 282mm utilizavel
    pdf.add_page()
    pdf.set_font('Helvetica', 'B', 16); pdf.set_text_color(44, 62, 80)
    pdf.cell(0, 15, 'RESUMO EXECUTIVO DO LAUDO PERICIAL MULTIPLO', 0, 1, 'C'); pdf.ln(5)
    pdf.set_fill_color(52, 73, 94); pdf.set_text_color(255, 255, 255)
    pdf.set_font('Helvetica', 'B', 10)
    pdf.cell(20,  10, 'Analise',              1, 0, 'C', fill=True)
    pdf.cell(115, 10, 'Assinatura Questionada (Binarizada)', 1, 0, 'C', fill=True)
    pdf.cell(40,  10, 'Z-Score',              1, 0, 'C', fill=True)
    pdf.cell(47,  10, 'Probabilidade',        1, 0, 'C', fill=True)
    pdf.cell(45,  10, 'Veredito',             1, 1, 'C', fill=True)
    pdf.set_text_color(0, 0, 0); pdf.set_font('Helvetica', '', 10)
    row_h = 35
    for i, res in enumerate(results_list):
        q_bin_path = os.path.join(save_dir, f"tmp_q_bin_{i}.png")
        y_start = pdf.get_y()
        # Paisagem: altura=210, margem inferior ~15, limite de quebra de pagina em 180
        if y_start + row_h > 180:
            pdf.add_page()
            pdf.set_fill_color(52, 73, 94); pdf.set_text_color(255, 255, 255)
            pdf.set_font('Helvetica', 'B', 10)
            pdf.cell(20,  10, 'Analise',              1, 0, 'C', fill=True)
            pdf.cell(115, 10, 'Assinatura Questionada (Binarizada)', 1, 0, 'C', fill=True)
            pdf.cell(40,  10, 'Z-Score',              1, 0, 'C', fill=True)
            pdf.cell(47,  10, 'Probabilidade',        1, 0, 'C', fill=True)
            pdf.cell(45,  10, 'Veredito',             1, 1, 'C', fill=True)
            pdf.set_text_color(0, 0, 0); pdf.set_font('Helvetica', '', 10)
            y_start = pdf.get_y()
        pdf.set_xy(15,  y_start); pdf.cell(20,  row_h, f"N.{i+1}", 1, 0, 'C')
        pdf.set_xy(35,  y_start); pdf.cell(115, row_h, "",        1, 0, 'C')
        pdf.image(q_bin_path, x=45, y=y_start+2.5, w=95)
        pdf.set_xy(150, y_start); pdf.cell(40,  row_h, f"{res['z_score']:.2f}", 1, 0, 'C')
        pdf.set_xy(190, y_start); pdf.cell(47,  row_h, f"{res['probability_percentage']:.1f}%", 1, 0, 'C')
        pdf.set_xy(237, y_start)
        p = res['probability_percentage']
        if p >= 70:   status = "AUTENTICA";     pdf.set_text_color(39, 174, 96)
        elif p >= 35: status = "INCONCLUSIVA";  pdf.set_text_color(243, 156, 18)
        else:         status = "FALSIDADE";     pdf.set_text_color(192, 57, 43)
        pdf.cell(45, row_h, status, 1, 1, 'C')
        pdf.set_text_color(0, 0, 0)
    pdf.ln(10)
    total = len(results_list)
    auth = sum(1 for r in results_list if r['probability_percentage'] >= 70)
    fake = sum(1 for r in results_list if r['probability_percentage'] < 35)
    inc  = total - auth - fake
    avg  = np.mean([r['probability_percentage'] for r in results_list])
    pdf.set_font('Helvetica', 'B', 14); pdf.set_text_color(44, 62, 80)
    pdf.cell(0, 10, 'CONCLUSAO ESTATISTICA GERAL DO LOTE PERICIADO:', 0, 1)
    pdf.set_font('Helvetica', '', 11); pdf.set_text_color(0, 0, 0)
    pdf.multi_cell(0, 6, enc(
        f"Total de documentos analisados: {total}  |  Autenticos: {auth}  |  Falsidades: {fake}  |  Inconclusivos: {inc}\n"
        f"Probabilidade Media do Lote: {avg:.2f}%"
    ))
    pdf.ln(4)
    if avg >= 70 and fake == 0:
        parecer = ("O conjunto documental analisado apresenta robusta e consolidada identidade caligrafica e geometrica "
                   "com a base de amostras autenticas. Conclui-se pela AUTENTICIDADE MAJORITARIA e convergente do lote.")
    elif avg < 35 or fake >= (total / 2):
        parecer = ("O conjunto repudia a identidade base em grande porte. A diferenca de tracados e perimetros "
                   "estruturais e drastica, sugerindo FALSIDADE ou severa tentativa de imitacao no corpus analisado.")
    elif auth > fake:
        parecer = ("O lote documental possui resultados mistos, porem inclinado a AUTENTICIDADE. Ha desvios em "
                   "assinaturas especificas que requerem cuidado, mas o predominio aponta para o mesmo autor.")
    else:
        parecer = ("O lote documental possui resultados mistos e alta divergencia interna. Impossivel atestar "
                   "unilateralidade para todo o conjunto — cada folha deve ser julgada individualmente.")
    pdf.set_font('Helvetica', 'B', 12)
    pdf.multi_cell(0, 6, enc(parecer))
    pdf.ln(20)
    pdf.set_font('Helvetica', 'B', 10)
    pdf.set_draw_color(44, 62, 80)
    pdf.line(60, pdf.get_y(), 237, pdf.get_y()); pdf.ln(5)
    pdf.cell(0, 5, "Assinatura e Carimbo do Perito Responsavel / Sistema GrafoAnalyzer PreciSion V2", 0, 1, 'C')

    # Cleanup temp images
    for t in temp_files:
        try: os.remove(t)
        except: pass

    if return_bytes:
        pdf_bytes = bytes(pdf.output())
        if _tmp_dir:
            try: os.rmdir(_tmp_dir)
            except: pass
        return pdf_bytes
    else:
        out_filename = f"Laudo_Multiplo_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        out_path = os.path.join(save_dir, out_filename)
        pdf.output(out_path)
        return out_path
