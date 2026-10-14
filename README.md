# தமிழ்நாடு வேளாண்மைத் திட்ட உதவியாளர் — RAG செயலி

Streamlit + LangChain + Hugging Face Embeddings + FAISS + OpenAI Chat Model பயன்படுத்தும் உள்ளக RAG செயலி. பயனர் உருவாக்கிய தமிழ்நாடு அரசு / வேளாண்மைத் திட்ட PDF ஆவணங்களைப் படித்து, அவற்றை உள்ளக வெக்டர் குறியீட்டில் சேமித்து, ஆதார PDF பெயர் மற்றும் பக்க எண்ணுடன் தமிழில் பதிலளிக்கிறது.

## முக்கிய அம்சங்கள்

- `.env` கோப்பிலிருந்து `OPENAI_API_KEY`, chat model, embedding model, PDF கோப்புறை, vector index கோப்புறை மற்றும் chunk/retrieval அமைப்புகளைப் படிக்கிறது.
- API key-ஐ UI-யில் காட்டவோ, source code-ல் hard-code செய்யவோாது.
- PDF கோப்புறையிலும் துணைக் கோப்புறைகளிலும் உள்ள PDF-களைத் தானாகக் கண்டறியும்.
- பல PDF-களை UI வழியாகவும் பதிவேற்றலாம்.
- Hugging Face `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` மாதிரி மூலம் தமிழ்/ஆங்கில வினாக்களுக்கு உள்ளக embeddings உருவாக்கப்படும்.
- FAISS index மற்றும் manifest உள்ளகமாக சேமிக்கப்படும்; கோப்புகள் / embedding model / chunk settings மாறாவிட்டால் குறியீட்டை மீண்டும் பயன்படுத்தும்.
- பதிலில் `[S1]` போன்ற citation-களும், கீழே PDF கோப்புப் பெயர், பக்க எண் மற்றும் தொடர்புடைய உரையும் காட்டப்படும்.
- UI தலைப்புகள், அமைப்புகள், உதாரண வினாக்கள், நிலை செய்திகள் மற்றும் பதில் வழிமுறைகள் தமிழில் உள்ளன.
- Streamlit file watcher முடக்கப்பட்டுள்ளது; இது உங்கள் Transformers installation-ல் முன்பு தெரிந்த `torchvision` optional-import traceback-ஐத் தவிர்க்க உதவும். Python கோப்புகளில் மாற்றம் செய்தால் செயலியை கைமுறையாக மறுதொடக்கம் செய்யவும்.

## தேவைகள்

- Windows-க்கு Python 3.12 பரிந்துரைக்கப்படுகிறது.
- முதல் முறை Hugging Face மாதிரியைப் பதிவிறக்க இணைய இணைப்பு தேவை. பின்னர் model cache-லிருந்து பயன்படுத்தப்படும்.
- பதில்களை உருவாக்க OpenAI API key மற்றும் API usage/billing access தேவை. ChatGPT subscription மட்டும் API credit வழங்காது.
- Hugging Face embeddings உள்ளகமாக இயங்கும்; அதற்காக OpenAI API key தேவையில்லை.

## Windows — நிறுவுதல்

1. ZIP-ஐ extract செய்து, திட்டக் கோப்புறை உள்ளே PowerShell திறக்கவும்.
2. தனி virtual environment உருவாக்கவும்:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

PowerShell activation தடுக்கப்பட்டால், அந்த PowerShell window-வில் மட்டும் `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` இயக்கி, environment-ஐ மீண்டும் activate செய்யவும்.

3. `.env.example`-ஐ `.env`-ஆக நகலெடுக்கவும்:

```powershell
Copy-Item .env.example .env
notepad .env
```

4. `.env`-இல் `OPENAI_API_KEY`-க்கு உங்கள் API key-ஐ இடவும். `PDF_FOLDER`-ஐ ஏற்கெனவே உருவாக்கிய திட்ட PDF-கள் இருக்கும் folder-ஆக மாற்றவும். உதாரணம்:

```dotenv
OPENAI_API_KEY=your-real-key-here
OPENAI_CHAT_MODEL=gpt-4o-mini
HF_EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
PDF_FOLDER=C:/Users/yourname/Documents/TamilNadu_Scheme_PDFs
RAG_INDEX_DIR=./.rag_index
CHUNK_SIZE=1000
CHUNK_OVERLAP=160
TOP_K=6
MAX_UPLOAD_MB=50
LOG_LEVEL=INFO
```

`PDF_FOLDER` மற்றும் `RAG_INDEX_DIR` relative paths ஆக இருந்தால் அவை இந்த project folder-ஐ அடிப்படையாகக் கொண்டு தீர்மானிக்கப்படும். `.env`-ஐ யாருடனும் பகிர வேண்டாம்; `.gitignore` அதைப் புறக்கணிக்கிறது. OS environment variable அமைக்கப்பட்டிருந்தால் அது `.env`-ஐ விட முன்னுரிமை பெறும்.

5. செயலியைத் தொடங்கவும்:

```powershell
python -m streamlit run app.py
```

Streamlit காட்டும் உள்ளக URL-ஐ browser-ல் திறக்கவும்; வழக்கமாக `http://localhost:8501`.

## பயன்பாடு

1. பக்கப்பட்டியில் திட்ட PDF folder சரியாக உள்ளதா என்பதை உறுதிப்படுத்தவும்.
2. **அறிவுத்தளத்தை உருவாக்கு / புதுப்பி** என்பதை அழுத்தவும்.
3. முதல் முறை Hugging Face மாதிரி download ஆகலாம்; indexing முடியும் வரை காத்திருக்கவும்.
4. தமிழ் அல்லது ஆங்கிலத்தில் திட்டத் தகுதி, நன்மைகள், மானியம், விண்ணப்ப முறை, தேவையான ஆவணங்கள் போன்ற கேள்விகளைக் கேட்கவும். உதவியாளர் இயல்பாக தமிழில் பதிலளிக்கும்.
5. புதிய PDF-களைச் சேர்த்தால் அல்லது `.env`-இல் embedding/chunk அமைப்புகளை மாற்றினால், index-ஐ மீண்டும் உருவாக்கவும். Embedding model மாறும்போது பழைய vectors-ஐ மறுபயன்படுத்த முடியாது.

### உதாரணக் கேள்விகள்

- இந்தத் திட்டத்திற்கு விண்ணப்பிக்க யாரெல்லாம் தகுதி பெறுகிறார்கள்?
- இந்தத் திட்டத்தின் கீழ் கிடைக்கும் மானியம் அல்லது நன்மைகள் என்ன?
- இந்தத் திட்டத்திற்கு எவ்வாறு விண்ணப்பிப்பது?
- விண்ணப்பிக்க என்னென்ன ஆவணங்கள் தேவை?
- இரண்டு திட்டங்களின் தகுதி நிபந்தனைகளை ஒப்பிடுங்கள்.

## அமைப்பு விவரங்கள்

- `config.py`: `.env` மற்றும் environment variables-ஐ load செய்து validate செய்கிறது.
- `app.py`: தமிழ் Streamlit UI, chat, upload, indexing controls மற்றும் source citations.
- `rag_core.py`: PDF extraction, chunking, Hugging Face embeddings, FAISS persistence மற்றும் index metadata.
- `.streamlit/config.toml`: Streamlit file watcher configuration.
- `tests/`: indexing utility-களுக்கான unit tests.

## வரம்புகள் மற்றும் பாதுகாப்பு

- தேர்ந்தெடுக்கப்பட்ட PDF-களிலிருந்து text extraction செய்யப்படும். Text இல்லாத scanned/image-only pages-க்கு OCR தனியாகச் செய்ய வேண்டும்.
- Embedding-கள் இந்த கணினியில் உருவாக்கப்படுகின்றன. பதிலை உருவாக்க கேள்வியும் மீட்டெடுக்கப்பட்ட PDF பகுதிகளும் OpenAI API-க்கு அனுப்பப்படும்.
- FAISS persistence-ல் pickle file உள்ளது. இந்தச் செயலி உருவாக்கிய, உங்களின் கட்டுப்பாட்டில் உள்ள index folder-ஐ மட்டுமே load செய்யவும்.
- இது ஆவண அடிப்படையிலான தகவல் உதவியாளர்; அதிகாரப்பூர்வத் தகுதி முடிவு அல்ல. முக்கியமான விவரங்களை சம்பந்தப்பட்ட அரசு துறையிடம் உறுதிப்படுத்தவும்.
- பொதுமக்கள் பயன்படுத்தும் production deployment-க்கு authentication, per-user access controls, rate limits, logging/monitoring, backup/restore மற்றும் concurrent index update control சேர்க்கப்பட வேண்டும்.
