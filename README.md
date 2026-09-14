# Ocena ergonomii stanowiska (RULA) — aplikacja w Pythonie

Aplikacja analizuje w czasie rzeczywistym obraz z kamery, wykrywa pozę
pracownika (YOLOv8-Pose), przelicza kąty ciała na punktację metody
**RULA** (Rapid Upper Limb Assessment) i pokazuje wynik na lokalnej
stronie WWW wraz z podglądem wideo, rozbiciem na poszczególne
segmenty ciała i wykresem historii oceny w czasie.

Skonfigurowana pod stanowisko pracy **stojącej**, z kamerą podłączoną
do MacBooka (np. Aurora 930 jako kamera USB/UVC lub wbudowana kamera
MacBooka).

## 1. Wymagania

- Python 3.10–3.12
- MacBook z chipem Apple M5 (aplikacja automatycznie użyje
  akceleracji GPU przez `mps`, jeśli dostępna w zainstalowanej wersji
  PyTorch; w innym wypadku działa na CPU)
- Kamera: wbudowana kamera MacBooka lub kamera Aurora 930 podłączona
  przez USB (system rozpoznaje ją jako standardową kamerę UVC — do
  tej aplikacji wykorzystywany jest tylko strumień obrazu RGB, bez
  danych głębi)

## 2. Instalacja

```bash
cd rula-ergonomia
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Model `yolov8n-pose.pt` zostanie pobrany automatycznie przy pierwszym
uruchomieniu (wymagany jednorazowy dostęp do internetu). Jeśli
komputer nie ma dostępu do sieci, plik modelu można pobrać wcześniej
ze strony Ultralytics i wskazać go zmienną środowiskową:

```bash
export RULA_YOLO_MODEL=/sciezka/do/yolov8n-pose.pt
```

## 3. Uruchomienie

```bash
python app.py
```

Następnie otwórz w przeglądarce: **http://localhost:8000**

> **Uwaga (macOS):** aplikacja domyślnie działa na porcie **8000**, a nie
> 5000 — port 5000 jest na macOS zajęty przez systemową usługę **AirPlay
> Receiver** (Ustawienia systemowe → Ogólne → AirDrop i Uchwyt), która w
> przeglądarce odpowiada stroną "Odmowa dostępu / HTTP ERROR 403" zamiast
> tej aplikacji. Port można zmienić zmienną środowiskową, np.
> `PORT=9000 python app.py`.

Aby otworzyć panel z innego urządzenia w tej samej sieci (np. telefonu),
użyj adresu IP MacBooka w sieci lokalnej, np. `http://192.168.2.154:8000`
(adres IP sprawdzisz w Ustawieniach systemowych → Sieć, lub poleceniem
`ipconfig getifaddr en0` w terminalu).

Jeśli podłączonych jest kilka kamer (np. wbudowana + Aurora 930),
wybierz właściwą z listy rozwijanej "Kamera" w panelu bocznym.
Zalecane ustawienie kamery: **z boku stanowiska, na wysokości bioder
pracownika**, tak aby dobrze widoczny był profil sylwetki — kąt
pochylenia tułowia i szyi liczony jest z rzutu 2D obrazu i jest
najdokładniejszy przy takim ustawieniu.

## 4. Jak liczony jest wynik RULA

Zaimplementowano oficjalne tabele RULA (McAtamney & Corlett, 1993):
Tabela A (ramię/przedramię/nadgarstek/skręt), Tabela B
(szyja/tułów/nogi) oraz Tabela C (wynik końcowy 1–7).

| Wynik końcowy | Interpretacja |
|---|---|
| 1–2 | Postawa akceptowalna |
| 3–4 | Wskazana dalsza obserwacja, mogą być potrzebne zmiany |
| 5–6 | Wskazana dalsza analiza, zmiany wkrótce |
| 7 | Wymagana natychmiastowa analiza i zmiany |

Kąty poszczególnych segmentów liczone są z 17 punktów
charakterystycznych zwracanych przez YOLOv8-Pose (`pose_utils.py`),
zgodnie ze standardowymi kryteriami kątowymi RULA (np. ramię: 0–20°
neutralne, 20–45° = 2 pkt, 45–90° = 3 pkt, >90° = 4 pkt, z korektami
za uniesiony/odwiedziony bark).

### Ograniczenia względem pełnej metody RULA

Poniższe elementy **nie są mierzalne** z obrazu pojedynczej kamery RGB
i estymacji pozy — aplikacja przyjmuje dla nich wartości domyślne,
które można ręcznie skorygować w panelu bocznym:

- **Nadgarstek i skręt nadgarstka** — wymagają obserwacji dłoni,
  której model pozy (17 punktów COCO) nie dostarcza. Przyjmowana jest
  wartość neutralna (1).
- **Siła / obciążenie (Force Score)** — RULA wymaga wiedzy o masie
  podnoszonego/trzymanego przedmiotu. Ustaw to ręcznie suwakiem
  "Ocena siły / obciążenia" zgodnie z rzeczywistą sytuacją na
  stanowisku.
- **Praca statyczna / powtarzalna (Muscle Use)** — aplikacja *stara
  się* wykrywać to automatycznie: jeśli zbliżona postawa utrzymuje się
  dłużej niż 60 sekund, dolicza +1 zgodnie z metodą RULA. To
  przybliżenie — nie zastępuje obserwacji charakteru pracy w dłuższym
  okresie.
- **Nogi (stanowisko stojące)** — aplikacja próbuje wykryć, czy ciężar
  ciała jest rozłożony symetrycznie na obie nogi na podstawie pozycji
  kostek względem bioder. Gdy detekcja jest niepewna (np. nogi poza
  kadrem), używana jest wartość z checkboxa "Zakładaj równomierne
  obciążenie nóg" (domyślnie: tak, czyli ocena 1).
- **Kąt tułowia/szyi przy kamerze ustawionej z przodu** — najlepiej
  odwzorowuje rzeczywiste zgięcie do przodu tylko przy kamerze
  patrzącej na osobę z boku (profil); przy kamerze czołowej wynik
  będzie bardziej odzwierciedlał przechył boczny.

**Ta aplikacja jest narzędziem pomocniczym/orientacyjnym i nie
zastępuje oceny wykonanej przez uprawnionego specjalistę ds. BHP /
ergonomii**, zwłaszcza przy podejmowaniu decyzji o zmianach
organizacyjnych na stanowisku pracy.

## 5. Dane i historia

- Ostatnie ~10 minut wyników (próbkowane co ok. 1 s) jest dostępne w
  interfejsie jako wykres.
- Pełna historia jest logowana do pliku `rula_log.csv` (co ok. 5 s) —
  można ją wykorzystać do dalszej analizy (np. w Excelu / pandas).

## 6. Struktura projektu

```
app.py            – serwer Flask, przechwytywanie obrazu, pętla YOLO+RULA, streaming MJPEG
rula.py           – tabele RULA (A/B/C) i funkcje kąt -> punktacja
pose_utils.py     – obliczanie kątów ciała z punktów charakterystycznych YOLOv8-Pose
templates/index.html – panel WWW (podgląd wideo, wynik, wykres, ustawienia)
requirements.txt  – zależności Pythona
rula_log.csv      – (tworzony automatycznie) log historii ocen
```

## 7. Typowe problemy

- **W konsoli widać `OpenCV: not authorized to capture video` / `can
  not spin main run loop from other thread`** — to najczęstszy
  problem na macOS: system pyta o zgodę na dostęp do kamery, ale robi
  to tylko z głównego wątku procesu. Aplikacja od tej wersji sama
  próbuje "rozgrzać" kamerę na starcie, żeby wywołać to okienko —
  **zaakceptuj je**, gdy się pojawi przy pierwszym uruchomieniu.
  Jeśli okienko się nie pojawiło (bo np. wcześniej odmówiono dostępu),
  napraw to ręcznie:
  1. Ustawienia systemowe → Prywatność i ochrona → Kamera → sprawdź,
     czy jest tam wpis dla Terminala / Pythona i czy jest zaznaczony.
  2. Jeśli wpisu nie ma albo dalej nie działa, zresetuj uprawnienie i
     spróbuj ponownie: `tccutil reset Camera` w terminalu, a potem
     uruchom `python app.py` jeszcze raz — okienko z prośbą powinno
     się pojawić.
- **"Nie mozna otworzyc kamery o indeksie N"** — na macOS upewnij się,
  że terminalowi / Pythonowi przyznano uprawnienia do kamery
  (Ustawienia systemowe → Prywatność i ochrona → Kamera).
- **Niska liczba klatek na sekundę** — domyślnie używany jest model
  `yolov8n-pose.pt` (najmniejszy, najszybszy). Dla lepszej dokładności
  kosztem szybkości można podać `RULA_YOLO_MODEL=yolov8s-pose.pt` lub
  większy wariant.
- **"Nie wykryto osoby"** — upewnij się, że górna część sylwetki
  (barki, biodra) mieści się w kadrze kamery.
