# 💻 GattoNero AI Assistant - Pseudokod Algorytmów

**Wersja:** 1.0  
**Cel:** Ten dokument zawiera implementację kluczowych algorytmów w formie pseudokodu. Stanowi on pomost między dokumentem koncepcyjnym a finalnym kodem w Pythonie.

---

## METODA 1: Palette Mapping (AUTO) - Basic

**Cel:** Przerysowanie obrazu docelowego przy użyciu ograniczonej palety barw wyekstrahowanej z obrazu wzorcowego.

```python
function palette_mapping(master_path, target_path, k=16, analysis_width=500):
	# 1. Wczytaj i przygotuj obraz MASTER
	master_img = cv2.imread(master_path)
	
	# 2. Zmniejsz obraz MASTER do analizy dla wydajności
	h, w, _ = master_img.shape
	scale = analysis_width / w
	small_master_img = cv2.resize(master_img, (analysis_width, int(h * scale)))
	
	# 3. Konwersja do RGB i przygotowanie danych dla K-Means
	master_rgb = cv2.cvtColor(small_master_img, cv2.COLOR_BGR2RGB)
	pixels = master_rgb.reshape((-1, 3))

	# 4. Znajdź dominującą paletę w obrazie MASTER
	kmeans = KMeans(n_clusters=k, n_init=10)
	kmeans.fit(pixels)
	master_palette = kmeans.cluster_centers_.astype('uint8')

	# 5. Wczytaj obraz TARGET w pełnej rozdzielczości
	target_img = cv2.imread(target_path)
	target_rgb = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
	
	# 6. Przygotuj dane z obrazu TARGET
	target_pixels = target_rgb.reshape((-1, 3))

	# 7. Dla każdego piksela w TARGET znajdź najbliższy kolor z palety MASTER
	labels = kmeans.predict(target_pixels)

	# 8. Stwórz nowy obraz na podstawie etykiet i palety
	result_pixels = master_palette[labels]
	result_img_rgb = result_pixels.reshape(target_img.shape)

	# 9. Konwertuj z powrotem do BGR do zapisu
	result_img_bgr = cv2.cvtColor(result_img_rgb, cv2.COLOR_RGB2BGR)

	return result_img_bgr
```

---

## METODA 2: Statistical Color Transfer (AUTO) - Professional

**Cel:** Przeniesienie "nastroju" kolorystycznego i tonalnego z obrazu master na target poprzez dopasowanie statystyk (średniej i odchylenia standardowego) w przestrzeni LAB.

```python
function statistical_transfer(master_path, target_path):
	# 1. Wczytaj obrazy
	master_img = cv2.imread(master_path)
	target_img = cv2.imread(target_path)

	# 2. Konwertuj obrazy do przestrzeni LAB i typu float64 dla precyzji
	master_lab = cv2.cvtColor(master_img, cv2.COLOR_BGR2LAB).astype("float64")
	target_lab = cv2.cvtColor(target_img, cv2.COLOR_BGR2LAB).astype("float64")

	# 3. Rozdziel kanały L, A, B dla obu obrazów
	L_master, A_master, B_master = cv2.split(master_lab)
	L_target, A_target, B_target = cv2.split(target_lab)

	# 4. Oblicz statystyki (średnia, odchylenie standardowe) dla każdego kanału
	# -- dla MASTER
	mu_L_master, std_L_master = L_master.mean(), L_master.std()
	mu_A_master, std_A_master = A_master.mean(), A_master.std()
	mu_B_master, std_B_master = B_master.mean(), B_master.std()
	# -- dla TARGET
	mu_L_target, std_L_target = L_target.mean(), L_target.std()
	mu_A_target, std_A_target = A_target.mean(), A_target.std()
	mu_B_target, std_B_target = B_target.mean(), B_target.std()

	# 5. Zastosuj transformację statystyczną dla każdego kanału osobno
	epsilon = 1e-6 # Zabezpieczenie przed dzieleniem przez zero
	
	# -- Kanał L (Jasność)
	L_new = (L_target - mu_L_target) * (std_L_master / (std_L_target + epsilon)) + mu_L_master
	
	# -- Kanał A (Zieleń-Czerwień)
	A_new = (A_target - mu_A_target) * (std_A_master / (std_A_target + epsilon)) + mu_A_master
	
	# -- Kanał B (Niebieski-Żółty)
	B_new = (B_target - mu_B_target) * (std_B_master / (std_B_target + epsilon)) + mu_B_master

	# 6. Przytnij wartości do prawidłowych zakresów dla przestrzeni LAB
	L_new = np.clip(L_new, 0, 100)
	A_new = np.clip(A_new, -128, 127)
	B_new = np.clip(B_new, -128, 127)

	# 7. Złącz przetransformowane kanały z powrotem w jeden obraz LAB
	result_lab = cv2.merge([L_new, A_new, B_new])

	# 8. Konwertuj obraz wynikowy do typu uint8 i przestrzeni BGR
	result_lab_uint8 = result_lab.astype("uint8")
	result_img_bgr = cv2.cvtColor(result_lab_uint8, cv2.COLOR_LAB2BGR)
	
	return result_img_bgr
```

---

## METODA 3: Histogram Matching (AUTO) - Exposure

**Cel:** Dopasowanie rozkładu tonalnego obrazu target do master. Najlepiej sprawdza się do wyrównywania ekspozycji i kontrastu.

```python
# Ta funkcja jest pomocnicza i będzie wywoływana dla każdego kanału
function match_channel_histogram(source_channel, template_channel):
	# 1. Oblicz histogramy i skumulowane funkcje dystrybucji (CDF)
	source_hist = cv2.calcHist([source_channel], [0], None, [256], [0, 256])
	template_hist = cv2.calcHist([template_channel], [0], None, [256], [0, 256])
	
	source_cdf = source_hist.cumsum()
	template_cdf = template_hist.cumsum()

	# 2. Normalizuj CDF
	source_cdf_norm = (source_cdf - source_cdf.min()) * 255 / (source_cdf.max() - source_cdf.min())
	template_cdf_norm = (template_cdf - template_cdf.min()) * 255 / (template_cdf.max() - template_cdf.min())

	# 3. Stwórz tablicę przyporządkowania (Lookup Table - LUT)
	lookup_table = np.zeros(256, dtype='uint8')
	g = 0
	for f in range(256):
		while g < 256 and template_cdf_norm[g] < source_cdf_norm[f]:
			g += 1
		lookup_table[f] = g

	# 4. Zastosuj LUT do kanału źródłowego
	return cv2.LUT(source_channel, lookup_table)


function histogram_matching(master_path, target_path):
	# 1. Wczytaj obrazy
	master_img = cv2.imread(master_path)
	target_img = cv2.imread(target_path)
	
	# 2. Konwertuj do przestrzeni LAB dla najlepszych rezultatów
	master_lab = cv2.cvtColor(master_img, cv2.COLOR_BGR2LAB)
	target_lab = cv2.cvtColor(target_img, cv2.COLOR_BGR2LAB)

	# 3. Rozdziel kanały
	L_master, A_master, B_master = cv2.split(master_lab)
	L_target, A_target, B_target = cv2.split(target_lab)
	
	# 4. Dopasuj histogram TYLKO dla kanału L (jasność)
	L_new = match_channel_histogram(L_target, L_master)

	# 5. Złącz nowy kanał L z oryginalnymi kanałami A i B
	result_lab = cv2.merge([L_new, A_target, B_target])
	
	# 6. Konwertuj z powrotem do BGR
	result_img_bgr = cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)

	return result_img_bgr
```

---

## METODA 6: ACES Color Space Transfer (CINEMATIC)

**Cel:** Profesjonalne dopasowanie kolorów w przestrzeni ACES dla kinematograficznej jakości.

```python
function aces_color_transfer(master_path, target_path):
	# 1. Wczytaj obrazy
	master_img = cv2.imread(master_path)
	target_img = cv2.imread(target_path)
	
	# 2. Konwertuj z sRGB do ACES2065-1 przez XYZ
	master_aces = srgb_to_aces2065(master_img)
	target_aces = srgb_to_aces2065(target_img)
	
	# 3. Konwertuj do ACEScct (logarytmiczna przestrzeń robocza)
	master_acescct = aces2065_to_acescct(master_aces)
	target_acescct = aces2065_to_acescct(target_aces)
	
	# 4. Zastosuj transformację statystyczną w ACEScct
	result_acescct = statistical_transfer_aces(master_acescct, target_acescct)
	
	# 5. Konwertuj z powrotem przez ACES2065-1 do sRGB
	result_aces = acescct_to_aces2065(result_acescct)
	result_srgb = aces2065_to_srgb_with_tonemapping(result_aces)
	
	return result_srgb

function srgb_to_aces2065(img):
	# Konwersja sRGB -> XYZ -> ACES2065-1
	img_linear = srgb_to_linear(img)
	xyz = linear_srgb_to_xyz(img_linear)
	aces = xyz_to_aces2065(xyz)
	return aces

function statistical_transfer_aces(master_acescct, target_acescct):
	# Podobnie jak w metodzie 2, ale w przestrzeni ACEScct
	for channel in [0, 1, 2]:  # R, G, B w ACEScct
		mu_master = master_acescct[:,:,channel].mean()
		std_master = master_acescct[:,:,channel].std()
		mu_target = target_acescct[:,:,channel].mean()
		std_target = target_acescct[:,:,channel].std()
		
		# Transformacja statystyczna
		epsilon = 1e-6
		target_acescct[:,:,channel] = (target_acescct[:,:,channel] - mu_target) * \
									  (std_master / (std_target + epsilon)) + mu_master
	
	return target_acescct
```

---

## METODA 7: Perceptual Color Matching (CIEDE2000)

**Cel:** Najdokładniejsze dopasowanie percepcyjne z użyciem metryki CIEDE2000.

```python
function ciede2000_color_matching(master_path, target_path, k=16):
	# 1. Wczytaj i przygotuj obrazy
	master_img = cv2.imread(master_path)
	target_img = cv2.imread(target_path)
	
	# 2. Stwórz paletę z master używając K-means w LAB
	master_lab = cv2.cvtColor(master_img, cv2.COLOR_BGR2LAB).astype('float64')
	master_pixels = master_lab.reshape((-1, 3))
	kmeans = KMeans(n_clusters=k)
	kmeans.fit(master_pixels)
	master_palette_lab = kmeans.cluster_centers_
	
	# 3. Konwertuj target do LAB
	target_lab = cv2.cvtColor(target_img, cv2.COLOR_BGR2LAB).astype('float64')
	target_pixels = target_lab.reshape((-1, 3))
	
	# 4. Dla każdego piksela znajdź najbliższy kolor używając CIEDE2000
	result_pixels = np.zeros_like(target_pixels)
	for i, pixel in enumerate(target_pixels):
		best_color_idx = find_closest_color_ciede2000(pixel, master_palette_lab)
		result_pixels[i] = master_palette_lab[best_color_idx]
	
	# 5. Rekonstruuj obraz i konwertuj do BGR
	result_lab = result_pixels.reshape(target_img.shape)
	result_bgr = cv2.cvtColor(result_lab.astype('uint8'), cv2.COLOR_LAB2BGR)
	
	return result_bgr

function find_closest_color_ciede2000(pixel_lab, palette_lab):
	# Implementacja metryki CIEDE2000
	min_distance = float('inf')
	best_idx = 0
	
	for i, palette_color in enumerate(palette_lab):
		distance = calculate_ciede2000(pixel_lab, palette_color)
		if distance < min_distance:
			min_distance = distance
			best_idx = i
	
	return best_idx

function calculate_ciede2000(lab1, lab2):
	# Implementacja pełnej formuły CIEDE2000
	# (skomplikowana formuła - wymagana biblioteka colorspacious lub własna implementacja)
	L1, a1, b1 = lab1[0], lab1[1], lab1[2]
	L2, a2, b2 = lab2[0], lab2[1], lab2[2]
	
	# Uproszczona wersja - w rzeczywistości CIEDE2000 jest znacznie bardziej złożona
	# Pełna implementacja wymaga ~50 linii kodu z wieloma poprawkami
	delta_L = L1 - L2
	delta_a = a1 - a2
	delta_b = b1 - b2
	
	# To jest tylko przybliżenie - prawdziwa CIEDE2000 uwzględnia:
	# - poprawki dla małych różnic
	# - nieliniowości w obszarze niebieskim
	# - kompensację dla neutralnych kolorów
	return np.sqrt(delta_L**2 + delta_a**2 + delta_b**2)  # Placeholder
```

---

## METODA 8: Adaptive Region-Based Matching (AI-ENHANCED)

**Cel:** Inteligentne dopasowanie z uwzględnieniem semantyki obrazu.

```python
function adaptive_region_matching(master_path, target_path):
	# 1. Wczytaj obrazy
	master_img = cv2.imread(master_path)
	target_img = cv2.imread(target_path)
	
	# 2. Segmentacja semantyczna target
	skin_mask, sky_mask, vegetation_mask, neutral_mask = segment_image_regions(target_img)
	
	# 3. Przygotuj obrazy w LAB
	master_lab = cv2.cvtColor(master_img, cv2.COLOR_BGR2LAB).astype('float64')
	target_lab = cv2.cvtColor(target_img, cv2.COLOR_BGR2LAB).astype('float64')
	
	# 4. Zastosuj dedykowane algorytmy dla każdego regionu
	result_lab = target_lab.copy()
	
	# Skóra - priorytet dla odcienia i nasycenia
	if np.any(skin_mask):
		result_lab = apply_skin_color_transfer(master_lab, result_lab, skin_mask)
	
	# Niebo - focus na odcieniu
	if np.any(sky_mask):
		result_lab = apply_sky_color_transfer(master_lab, result_lab, sky_mask)
	
	# Roślinność - zachowanie naturalnych zieleni
	if np.any(vegetation_mask):
		result_lab = apply_vegetation_color_transfer(master_lab, result_lab, vegetation_mask)
	
	# Neutralne obszary - standardowe dopasowanie
	if np.any(neutral_mask):
		result_lab = apply_statistical_transfer_masked(master_lab, result_lab, neutral_mask)
	
	# 5. Wygładź granice między regionami
	result_lab = smooth_region_boundaries(result_lab, [skin_mask, sky_mask, vegetation_mask, neutral_mask])
	
	# 6. Konwertuj do BGR
	result_bgr = cv2.cvtColor(result_lab.astype('uint8'), cv2.COLOR_LAB2BGR)
	
	return result_bgr

function segment_image_regions(img):
	# Prosta segmentacja oparta na kolorach HSV
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	
	# Maska skóry (odcienie pomarańczowo-różowe)
	skin_mask = detect_skin_regions(hsv)
	
	# Maska nieba (odcienie niebieskie, wysoka jasność)
	sky_mask = detect_sky_regions(hsv)
	
	# Maska roślinności (odcienie zielone)
	vegetation_mask = detect_vegetation_regions(hsv)
	
	# Pozostałe obszary
	neutral_mask = ~(skin_mask | sky_mask | vegetation_mask)
	
	return skin_mask, sky_mask, vegetation_mask, neutral_mask

function apply_skin_color_transfer(master_lab, target_lab, mask):
	# Specjalne dopasowanie dla skóry - zachowanie naturalnych tonów
	# Focus na kanałach a* i b* (chromatyczność) z delikatną modyfikacją L*
	masked_master = master_lab[mask]
	masked_target = target_lab[mask]
	
	# Oblicz statystyki tylko dla kanałów a* i b*
	for channel in [1, 2]:  # a*, b*
		mu_master = masked_master[:, channel].mean()
		std_master = masked_master[:, channel].std()
		mu_target = masked_target[:, channel].mean()
		std_target = masked_target[:, channel].std()
		
		# Delikatna transformacja (50% siły)
		epsilon = 1e-6
		transfer_strength = 0.5
		new_values = (masked_target[:, channel] - mu_target) * \
					 (std_master / (std_target + epsilon)) + mu_master
		target_lab[mask, channel] = masked_target[:, channel] * (1 - transfer_strength) + \
								new_values * transfer_strength
	
	return target_lab
```

---

## METODA 9: Temporal Consistency (VIDEO)

**Cel:** Spójność kolorystyczna w sekwencjach wideo z eliminacją flickeringu.

```python
function temporal_color_matching(master_path, target_sequence_paths, temporal_window=5):
	# 1. Wczytaj obraz wzorcowy
	master_img = cv2.imread(master_path)
	master_lab = cv2.cvtColor(master_img, cv2.COLOR_BGR2LAB).astype('float64')
	
	# 2. Wczytaj sekwencję klatek
	frames = []
	for path in target_sequence_paths:
		frame = cv2.imread(path)
		frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype('float64'))
	
	# 3. Pierwsza klątka - standardowe dopasowanie
	result_frames = []
	result_frames.append(statistical_transfer(master_lab, frames[0]))
	
	# 4. Kolejne klatki - z uwzględnieniem temporal consistency
	for i in range(1, len(frames)):
		# Oblicz dopasowanie dla bieżącej klatki
		current_matched = statistical_transfer(master_lab, frames[i])
		
		# Zastosuj temporal smoothing
		window_start = max(0, i - temporal_window)
		window_frames = result_frames[window_start:i]
		
		# Wygładź względem poprzednich klatek
		smoothed_frame = apply_temporal_smoothing(current_matched, window_frames)
		
		# Dodatkowa redukcja flickeringu
		flicker_reduced = reduce_temporal_flicker(smoothed_frame, result_frames[i-1])
		
		result_frames.append(flicker_reduced)
	
	# 5. Konwertuj wszystkie klatki z powrotem do BGR
	result_bgr_frames = []
	for frame_lab in result_frames:
		frame_bgr = cv2.cvtColor(frame_lab.astype('uint8'), cv2.COLOR_LAB2BGR)
		result_bgr_frames.append(frame_bgr)
	
	return result_bgr_frames

function apply_temporal_smoothing(current_frame, previous_frames, alpha=0.3):
	# Exponential moving average dla wygładzenia czasowego
	if len(previous_frames) == 0:
		return current_frame
	
	# Oblicz średnią ważoną z poprzednich klatek
	weighted_history = np.zeros_like(current_frame)
	total_weight = 0
	
	for i, prev_frame in enumerate(reversed(previous_frames)):
		weight = alpha ** (i + 1)
		weighted_history += prev_frame * weight
		total_weight += weight
	
	weighted_history /= total_weight
	
	# Miksuj z bieżącą klatką
	temporal_strength = 0.2  # 20% wpływu historii
	result = current_frame * (1 - temporal_strength) + weighted_history * temporal_strength
	
	return result

function reduce_temporal_flicker(current_frame, previous_frame, threshold=5.0):
	# Redukuj nagłe zmiany kolorów między klatkami
	diff = np.abs(current_frame - previous_frame)
	
	# Znajdź obszary z dużymi zmianami
	flicker_mask = np.any(diff > threshold, axis=2)
	
	# W obszarach flickeringu zastosuj łagodniejsze przejście
	result = current_frame.copy()
	blend_factor = 0.7  # 70% nowej klatki, 30% poprzedniej
	
	result[flicker_mask] = current_frame[flicker_mask] * blend_factor + \
						  previous_frame[flicker_mask] * (1 - blend_factor)
	
	return result
```