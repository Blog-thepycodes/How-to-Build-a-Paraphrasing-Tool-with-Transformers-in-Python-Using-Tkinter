import threading
import tkinter as tk
from tkinter import ttk, scrolledtext
from transformers import PegasusForConditionalGeneration, PegasusTokenizerFast, AutoModelForSeq2SeqLM, AutoTokenizer


# Load models (deferred to when selected)
pegasus_model = None
pegasus_tokenizer = None
flan_t5_model = None
flan_t5_tokenizer = None


# Function to load the selected model dynamically
def load_model(selected_model):
   global pegasus_model, pegasus_tokenizer, flan_t5_model, flan_t5_tokenizer
   status_label.config(text="Loading model, please wait...")


   try:
       if selected_model == "PEGASUS" and pegasus_model is None:
           pegasus_model = PegasusForConditionalGeneration.from_pretrained("tuner007/pegasus_paraphrase")
           pegasus_tokenizer = PegasusTokenizerFast.from_pretrained("tuner007/pegasus_paraphrase")
       elif selected_model == "FLAN-T5" and flan_t5_model is None:
           flan_t5_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
           flan_t5_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
   except Exception as e:
       output_text.delete("1.0", tk.END)
       output_text.insert(tk.END, f"Error loading model: {str(e)}")
       return False


   status_label.config(text="Model loaded.")
   return True


# Function to get paraphrased sentences
def get_paraphrased_sentences(model, tokenizer, sentence, num_return_sequences=5, num_beams=5):
   inputs = tokenizer([sentence], truncation=True, padding="longest", return_tensors="pt")
   outputs = model.generate(
       **inputs,
       num_beams=num_beams,
       num_return_sequences=num_return_sequences,
   )
   return tokenizer.batch_decode(outputs, skip_special_tokens=True)


# Function to handle the paraphrasing in a separate thread
def paraphrase_text():
   selected_model = model_choice.get()
   sentence = input_text.get("1.0", tk.END).strip()


   if not sentence:
       output_text.delete("1.0", tk.END)
       output_text.insert(tk.END, "Please enter a sentence.")
       return


   paraphrased_sentences = []


   if not load_model(selected_model):
       return


   status_label.config(text="Paraphrasing, please wait...")
   paraphrase_button.config(state=tk.DISABLED)


   try:
       if selected_model == "PEGASUS":
           paraphrased_sentences = get_paraphrased_sentences(pegasus_model, pegasus_tokenizer, sentence)
       elif selected_model == "FLAN-T5":
           paraphrased_sentences = get_paraphrased_sentences(flan_t5_model, flan_t5_tokenizer, sentence)
   except Exception as e:
       paraphrased_sentences = [f"Error during paraphrasing: {str(e)}"]


   # Display paraphrased sentences
   output_text.delete("1.0", tk.END)
   for idx, paraphrase in enumerate(paraphrased_sentences):
       output_text.insert(tk.END, f"{idx + 1}. {paraphrase}\n\n")


   status_label.config(text="Paraphrasing complete.")
   paraphrase_button.config(state=tk.NORMAL)


# Function to run paraphrasing in a new thread to avoid freezing the UI
def run_paraphrasing():
   threading.Thread(target=paraphrase_text).start()


# Tkinter window setup
window = tk.Tk()
window.title("Paraphrasing Tool with Transformers - The Pycodes")
window.geometry("700x550")
window.resizable(True, True)


# Label for input
input_label = tk.Label(window, text="Enter text to paraphrase:")
input_label.pack(pady=10)


# Input text box
input_text = scrolledtext.ScrolledText(window, wrap=tk.WORD, width=60, height=5)
input_text.pack(pady=10)


# Dropdown menu to select model
model_label = tk.Label(window, text="Select model:")
model_label.pack(pady=10)


# Default model is PEGASUS
model_choice = ttk.Combobox(window, values=["PEGASUS", "FLAN-T5"])
model_choice.current(0)  # Set PEGASUS as default
model_choice.pack()


# Button to trigger paraphrasing
paraphrase_button = tk.Button(window, text="Paraphrase", command=run_paraphrasing)
paraphrase_button.pack(pady=10)


# Output text box
output_label = tk.Label(window, text="Paraphrased Text:")
output_label.pack(pady=10)


output_text = scrolledtext.ScrolledText(window, wrap=tk.WORD, width=60, height=10)
output_text.pack(pady=10)


# Status label
status_label = tk.Label(window, text="Waiting for input...", fg="blue")
status_label.pack(pady=10)


# Start the Tkinter loop
window.mainloop()
