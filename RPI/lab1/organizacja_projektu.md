# Organizacja i infrastruktura projektu

## 1. Opis projektu i produktu
*   **Nazwa projektu:** System przetwarzania danych oparty na pojedynczym oraz współpracujących modelach LLM
*   **Adresowany problem:** Badanie efektywności kolektywnego działania agentów LLM ("Hive Mind") w porównaniu do wydajności pojedynczego modelu w rozwiązywaniu złożonych problemów matematycznych i logicznych.
*   **Obszar zastosowania:** Inżynieria systemów wieloagentowych, optymalizacja wnioskowania LLM, automatyczne rozwiązywanie problemów (Automated Problem Solving).
*   **Rynek:** Deweloperzy systemów AI, badacze zajmujący się orkiestracją modeli językowych.
*   **Interesariusze:** Opiekunowie projektu (PG WETI), zespół projektowy.
*   **Użytkownicy i ich potrzeby:** Programiści systemów AI szukający metod na zwiększenie rzetelności (reliability) odpowiedzi oraz zmniejszenie halucynowania modeli bez konieczności ich douczania.
*   **Cel i zakres produktu:** Implementacja i analiza systemu "Hive Mind" składającego się z agenta badawczego (Researcher), zestawu wyspecjalizowanych kalkulatorów (Algebra, Stepwise, Base) oraz surowego ewaluatora (Evaluator). Kluczowym celem jest weryfikacja, czy konsensus wypracowany przez grupę agentów przewyższa jakościowo odpowiedź pojedynczego modelu.
*   **Ograniczenia:** Aktualnie wykorzystanie lokalnej instancji Ollama (`llama3.1:8b`), ograniczona liczba iteracji obliczeniowych (CALCULATION_RUNS).
*   **Termin:** Termin składania prac inżynierskich na WETI.
*   **Główne etapy projektu:**
    1.  Opracowanie ról agentów (Researcher, Calculator, Evaluator).
    2.  Implementacja mechanizmu "Hive Mind" w Pythonie (folder `Pure/`).
    3.  Przygotowanie zestawu testowego (plik `questions.txt`).
    4. Rozszerzenie projektu o kolejne modele.
    4.  **Zadanie kluczowe:** Przeprowadzenie testów porównawczych między pojedynczym modelem a systemem Hive Mind.
    5.  Analiza statystyczna wyników i wnioski.

## 2. Interesariusze i użytkownicy
*   **Interesariusze:** Opiekunowie przedmiotu oraz zespół projektowy.
*   **Użytkownicy końcowi:** Osoby potrzebujące wysokiej precyzji w obliczeniach symbolicznych i numerycznych wykonywanych przez LLM.

## 3. Zespół
*   **Skład zespołu:** Jakub Nowak 197860, Oliwier Komorowski 197808.
*   **Podział ról:**
    *   Zadania przydzielane są dynamicznie w zależności od potrzeb projektu.
    *   Dwaj programiści.
*   **Umiejętności:** Python, analiza danych, Prompt Engineering, obsługa Ollama API.
*   **Praca:** Współpraca zespołowa nad wspólnym kodem.

## 4. Komunikacja w zespole i z interesariuszami
*   **Komunikacja wewnętrzna:** Spotkania grupy roboczej, wymiana spostrzeżeń dotyczących nowych implementacji (np. zachowania agentów w trakcie testów) oraz selekcja kolejnych zadań.
*   **Komunikacja zewnętrzna:** Konsultacje z opiekunem projektu w celu weryfikacji wprowadzanych zmian oraz obranego kierunku rozwoju projektu.

## 5. Współdzielenie dokumentów i kodu
*   **Kod źródłowy:** Repozytorium Git, aktualna i docelowa logika projektu znajduje się w folderze `Pure/`.
*   **Zarządzanie zmianami:** Wykorzystanie mechanizmów Git do śledzenia postępów w projekcie.
*   **Dokumentacja:** Pliki oraz raporty techniczne w folderze `Dokumentacja/`.

## 6. Narzędzia
*   **Środowisko:** Python 3, program PyCharm.
*   **Infrastruktura LLM:** Ollama (lokalny serwer API na porcie 11434).
*   **Modele:** Głównie `llama3.1:8b` (w planie wprowadzenie kolejnych).
*   **Testy:** Zbiór zadań matematycznych w `Pure/questions.txt`.
*   **Analiza performance'u:** Skrypty mierzące czas odpowiedzi, poprawność (accuracy) i liczbę iteracji potrzebnych do osiągnięcia konsensusu przez Ewaluatora.
