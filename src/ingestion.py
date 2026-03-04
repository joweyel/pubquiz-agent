import os

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader

# Transcribing of mp3 audios
from faster_whisper import WhisperModel

load_dotenv("../")

embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=1536)

def get_whisper_model():
    """Obtains a faster whisper model based on the hardware at hand.

    Returns:
        A configured Faster-Whisper model.
    """
    try:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_args = {
            "model_size_or_path": "small",
            "device": device,
            "compute_type": "float16" if device == "cuda" else "int8",
        }
    except ImportError:
        print("CPU-Fallback")
        model_args = {
            "model_size_or_path": "base",
            "device": "cpu",
            "compute_type": "int8",
        }
    return WhisperModel(**model_args)


def transcribe(audio_file_path: str, model: WhisperModel):
    """Function that loads audio files and transcribes them with
    faster Whisper model.

    Args:
        audio_file_path: An audio file's full path.
        model: Whisper model that is used to transcribe provided audio.

    Returns:
        The transcribed text from the audio file.
    """
    segments, _ = model.transcribe(audio_file_path, language="de")
    return "".join(segment.text for segment in segments)


def load_documents(text_data_dir: str) -> list[Document]:
    loader = DirectoryLoader(text_data_dir, glob="*.txt", loader_cls=TextLoader)
    return loader.load()


def ingest_data(text_data_dir: str, audio_data_dir: str):
    # ================================================================ #
    # Step 1: Transcribing and saving audio files to text data folder  #
    # ================================================================ #

    # Pre-processing audio (convert to text) for ingestion
    audio_files = [
        os.path.join("./", audio_data_dir, audio_file)
        for audio_file in os.listdir(audio_data_dir)
        if audio_file.endswith(".mp3")
    ]
    if audio_files:
        whisper_model = get_whisper_model()
        for i, audio_file in enumerate(audio_files):
            transcript_name = os.path.join(
                text_data_dir,
                audio_file.split("/")[-1] + ".txt",  #  audio_file_name.mp3.txt
            )
            if os.path.exists(transcript_name):
                print(f"[{i+1}] Skipping: {audio_file.split('/')[-1]}")
                continue
            print(f"[{i+1}] Tanscribing: {audio_file.split('/')[-1]}")

            with open(transcript_name, "w") as f:
                transcript = transcribe(audio_file, whisper_model)
                f.write(transcript)

    # ================================================================ #
    # Step 2: Reading all text files from the text data folder         #
    # ================================================================ #

    documents = load_documents(text_data_dir)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1_000,
        chunk_overlap=100,
    )
    splitted_docs = text_splitter.split_documents(documents)

    # ================================================================ #
    # Step 3: Storing the extracted docs in VectorDB for retrieval     #
    # ================================================================ #
    ids = [f"{doc.metadata.get('source', 'unknown')}::{i}" for i, doc in enumerate(splitted_docs)]
    vector_store = Chroma(
        persist_directory="./PubDatabase/chroma/", embedding_function=embeddings
    )
    vector_store.add_documents(splitted_docs, ids=ids)


if __name__ == "__main__":
    ingest_data("./PubTexts", "./PubAudio")