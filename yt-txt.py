import os
import subprocess
import whisper
import ssl
import openai

openai.api_key = "Your Api key" #Generate and use your api key

ssl._create_default_https_context = ssl._create_unverified_context

def download_audio(url, output_path):
    # path = os.path.join(output_path, "%(title)s.%(ext)s")
    path = os.path.join(output_path, "file.%(ext)s")
    command = [
        "youtube-dl",
        "-x",  # Extract audio
        "--audio-format", "mp3",
        "--output", path,
        url,
    ]
    # Check if the audio can be saved onto temp memory than saving each file
    subprocess.run(command, check=True)

def summarize_text(text,prompt):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=f"{prompt}\n{text}",
        temperature=0.3,
        max_tokens=600,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=1
    )

    summary = response.choices[0].text.strip()
    return summary


def main():
    ytURL = input("Enter YouTube video URL: ")
    output_dir = "Your path" #Add a path to where you want to store the files

    download_audio(ytURL, output_dir)
    print(f"Audio file saved to {output_dir}")
    
    model = whisper.load_model("base")

    # load the entire audio file
    audio = whisper.load_audio(os.path.join(output_dir, "file.mp3"))
    # Since its a CSV it doesn't need to save the name dynamically. 

    options = {
        "language": "en", # input language, if omitted is auto detected
        "task": "translate" # or "transcribe" if you just want transcription
    }
    
    result = whisper.transcribe(model, audio, **options)
    with open("output.txt", "w") as f:
        f.write(result["text"])
        
    summarization_prompt = input("Enter a prompt for summarization: ")
    summary = summarize_text(result["text"], summarization_prompt)
    with open("summary.txt", "w") as f:
        f.write(summary)

if __name__ == "__main__":
    main()