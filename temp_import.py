

def analyze_colors(image_rgb, step=5):
    sampled_img = image_rgb[::step, ::step]
    r, g, b = cv2.split(sampled_img)
    return r, g, b

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=7.0, tileGridSize=(8,8))
    contrast_enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(contrast_enhanced, (7,7), 0)
    _, binary = cv2.threshold(blurred, 155, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary

def segment_cells_improved(image, binary):
    kernel = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(cleaned, kernel, iterations=3)
    dist_transform = cv2.distanceTransform(cleaned, cv2.DIST_L2, 5)
    
    if np.max(dist_transform) > 0:
        dist_norm = cv2.normalize(dist_transform, None, 0, 255, cv2.NORM_MINMAX)
        dist_norm = np.uint8(dist_norm)
        _, sure_fg = cv2.threshold(dist_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        sure_fg = cleaned.copy()
    
    unknown = cv2.subtract(sure_bg, sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
    gray_enhanced = clahe.apply(gray)
    gradient = cv2.morphologyEx(gray_enhanced, cv2.MORPH_GRADIENT, kernel)
    markers = cv2.watershed(cv2.cvtColor(gradient, cv2.COLOR_GRAY2BGR), markers)
    
    return markers

def count_and_filter(image, markers, min_size=100, max_size=600):
    counts = 0
    areas = []
    for label in np.unique(markers):
        if label in [0, -1]:
            continue
        mask = np.zeros(markers.shape, dtype="uint8")
        mask[markers == label] = 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            areas.append(area)
            if min_size <= area <= max_size:
                counts += 1
    return counts, areas

def process_single_image(image_path, step=5):
    img = cv2.imread(image_path)
    if img is None:
        return None, None, None, None, None
    filename = os.path.basename(image_path)
    try:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        r, g, b = analyze_colors(img_rgb, step)
        binary = preprocess_image(img)
        markers = segment_cells_improved(img, binary)
        cell_count, areas = count_and_filter(img, markers)
        color_stats = {
            'filename': filename,
            'Red': np.mean(r),
            'Green': np.mean(g),
            'Blue': np.mean(b),
            'R/G': np.mean(r) / (np.mean(g) + 0.001),
            'R/B': np.mean(r) / (np.mean(b) + 0.001),
            'G/B': np.mean(g) / (np.mean(b) + 0.001),
            'pixels_sampled': r.size
        }
        cell_stats = {
            'Cell Count': cell_count,
            'Mean Area': np.mean(areas) if areas else 0,
            'Median Area': np.median(areas) if areas else 0,
            'Min Area': min(areas) if areas else 0,
            'Max Area': max(areas) if areas else 0,
            'Areas': areas
        }
        return color_stats, cell_stats, r, g, b
    except Exception:
        return None, None, None, None, None
