# 🧠 Vakyam AI (Offline AI)

**An offline duo-lingual RAG assistant for intelligent document understanding**, that can:

• Read PDFs using **OCR**
• Answer questions from the document
• Generate **summaries**
• Support **Hindi & English**
• Provide **audio responses**

The system runs **completely offline using Llama models**.

---

# ✨ Features

* 📄 **PDF understanding with OCR (Hindi supported)**
* 🔎 **Semantic search using FAISS**
* 🤖 **Local LLM inference using Llama.cpp**
* 🗣 **Audio responses with gTTS**
* 🌍 **Hindi + English support**
* ⚡ **Optimized CPU inference**
* 🧠 **Query correction using difflib**
* 📚 **Document grounded answers with sources**

---

# 🧩 Architecture

```
PDF
 ↓
OCR (Tesseract)
 ↓
Text Chunking
 ↓
SentenceTransformer Embeddings
 ↓
FAISS Vector Search
 ↓
Llama-3 LLM
 ↓
Answer / Summary
 ↓
Text + Audio Output
```

---

# 📦 Installation

Clone the repo

```
git clone https://github.com/Kunjalgarg/Vakyam-RAG.git
cd Vakyam-RAG
```

Create virtual environment

```
python -m venv venv
venv\Scripts\activate
```

Install dependencies

```
pip install -r requirements.txt
```

---

# 🧠 Download Model

Download a GGUF model (example):

Meta Llama 3 Instruct

https://huggingface.co

Place it in the project folder.

Example:

```
Meta-Llama-3-8B-Instruct-Q4_K_M.gguf
```

---

# ▶️ Run the Assistant

```
python main.py
```

---

# 💻 Example Terminal Output

```
=====================================
        PDF RAG Assistant
=====================================


Choose:
1 → Summary
2 → Q/A
3 → Both
Enter (1/2/3): 1

Language: E (English) / H (Hindi): h

Output:        
1 → Text       
2 → Audio      
3 → Both       
Enter (1/2/3): 3

============================================================
SUMMARY
============================================================
यह कहानी "कफ़न" के नाम से जाता है, जिसमें एक गरीब चमार परिवार की दुर्दशा को प्रस्तुत किया गया है। माधव और उसके बाप, घीसू दोनों एक अलाव के सामने ख़ामोश बैठे हुए थे, जबकि माधव की नौजवान पत्नी, बुधिया, दर्द-ए-ज़ेह से पछ्छाड़ें खा रही थी। जाड़ों की रात थी और फ़ज़ा सन्नाटे में ग़र्क़ हो गया था। घीसू ने कहा कि बुधिया बचेगी नहीं और उन्होंने सुझाव दिया कि वह जल्दी मर जाए। माधव ने दर्दनाक लहजे में जवाब दिया कि अगर मरना है तो जल्दी मर क्यूँ नहीं जाती। बुधिया की बेवफाई से उन्होंने अपना गुस्सा प्रकट किया। चमार परिवार की दुर्दशा को प्रस्तुत किया गया है, जहाँ माधव और घीसू दोनों कामचोर थे। वे एक दिन काम करते तो तीन दिन आराम, जबकि माधव इतना कामचोर था कि घंटे भर काम करता तो घंटे भर चिलम पीता। इसलिए उन्हें कोई रखता ही नहीं था। घर में मुट्ठी भर अनाज भी मौजूद होता था, लेकिन उनके लिए काम करने की क़सम थी। जब दो एक फ़ाक़े हो जाते तो वे लकड़ी तोड़ते या कोई मजदूरी तलाश करते। गाँव में काम की कमी न थी, लेकिन उन्हें लोग उसी वक़्त बुलाते जब दो आदमियों से एक का काम पा कर भी क़नाअत कर लेने के सिवा और कोई चारा न होता। कहानी में अजीब ज़िंदगी की झलक दिखाई गई है, जहाँ घर में मिट्टी के दो चार बर्तनों के सिवा कोई असासा नहीं था। वे फटे चीथड़ों से अपनी उर्यानी को ढाँके हुए दुनिया की फ़िक्रों से आज़ाद थे। उनकी जिंदगी में कोई ग़म नहीं था। कहानी का अंतिम पन्ना पर, घीसू ने आलुओं का बड़ा हिस्सा साफ़ कर देने की बात कही, लेकिन माधव ने इसका जवाब नहीं दिया। यहाँ तक कि वो 
दोनों अलाव के सामने बैठे हुए थे, जिसका मतलब था कि उनकी पत्नियां मर चुकीं, और अब वो दोनों शायद इसी इंतिज़ार में हैं कि वो मर जाए तो आराम से सोएँ। इस कहानी में गरीबी, बेरोजगारी और परिवार की समस्याओं को दर्शाया गया है।    
============================================================

Audio saved → audio_outputs\audio_0a3ce4c67bce4927a9744c202a06d7f7.mp3

Your Question (type exit to quit): घीसू कौन था?

घीसू चमारों का एक सदस्य था, जो मेहनती नहीं था और सिर्फ़ आराम करता रहता था। वह न तो अपने हाथ-पाँव पटकना देखता था और न ही कड़ी मेहनत कर्ता था। उसके लिए को
ई रखता भी नहीं, इसलिए उसे घर में मुट्ठी भर अनाज भी मौजूद रहता था, जिसके बदले वह लकड़ियाँ तोड़ता और बेचता था।

Your Question (type exit to quit): घीसू, माधव और बुधिया में से कौन अच्छा था?

Audio saved → audio_outputs\voice_2519d0b112e24365aeb0223475318b94.mp3

Your Question (type exit to quit): exit

See you soon 😉

```

# 📚 Technologies Used

* **Llama.cpp**
* **SentenceTransformers**
* **FAISS**
* **Tesseract OCR**
* **gTTS**
* **Python**

---

# 🚀 Future Improvements

* Voice input
* Multi-lingual support
* Streaming responses
* Web/App UI
* Multi-PDF support
* Semantic caching

---

# 👨‍💻 Author

Developed by **Kunjal Garg**

AI / NLP Projects
