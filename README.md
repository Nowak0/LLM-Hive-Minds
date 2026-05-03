# LLM Hive Minds

The aim of the project is the development of a distributed system enabling multiple LLMs to collaborate simultaneously in a way inspired by the concept of hive minds.
The system focuses on improvement of coherence and contextual accuracy of generated responses. 

Currently, our system is designed to handle _mathematical_ questions.

The project is realized as a part of a "Team Project" course as well as two engineering diploma projects at Gdańsk University of Technology

## Running

To use the app clone this repository to your device and run these commands in the main folder:

### User input questions

     main.py

### Chosen questions from the question bank

    main.py qBank

## Technology Stack

* [__Ollama__](https://ollama.com/) - allows for the usage of local LLMs
  * [__Llama3.1:8b__](https://ollama.com/library/llama3.1:8b) - main LLM used during the development 