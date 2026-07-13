# import os
# import chromadb
# from pypdf import PdfReader
# from langchain_text_splitters import RecursiveCharacterTextSplitter

# # 1. Local ChromaDB එක හාඩ් එකේ සේව් වෙන්න සෙටප් කිරීම
# client = chromadb.PersistentClient(path="./chroma_db")
# collection = client.get_or_create_collection(name="financial_docs")

# # 2. Text කෑලි වලට කඩන Splitter එක
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# # 3. PDF දාන්න ඕන ෆෝල්ඩර් එක සෑදීම
# pdf_folder = "./my_pdfs"
# if not os.path.exists(pdf_folder):
#     os.makedirs(pdf_folder)
#     print(f"📁 '{pdf_folder}' කියලා ෆෝල්ඩර් එකක් හැදුවා. ඒක ඇතුළට ඔයාගේ PDF 20-30 දාලා ආයෙත් රන් කරන්න.")
#     exit()

# pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith('.pdf')]

# if not pdf_files:
#     print("⚠️ './my_pdfs' ෆෝල්ඩර් එක ඇතුළේ කිසිම PDF එකක් නැහැ! කරුණාකර PDF එකක් දමා නැවත උත්සාහ කරන්න.")
#     exit()

# id_counter = 0
# for pdf in pdf_files:
#     pdf_path = os.path.join(pdf_folder, pdf)
#     print(f"📖 Reading document: {pdf}")
#     reader = PdfReader(pdf_path)
    
#     for page_num, page in enumerate(reader.pages):
#         text = page.extract_text()
#         if not text:
#             continue
            
#         chunks = text_splitter.split_text(text)
        
#         for chunk in chunks:
#             # ChromaDB මඟින් ඔයාගේ කම්පියුටරය ඇතුළේම නොමිලේම Embedding හදලා සේව් කරගන්නවා
#             collection.add(
#                 documents=[chunk],
#                 metadatas=[{"source": pdf, "page": page_num + 1}],
#                 ids=[f"id_{id_counter}"]
#             )
#             id_counter += 1

# print(f"✅ සාර්ථකයි! මුළු කෑලි (Chunks) {id_counter}ක් ඔයාගේ Local පරිගණකයේ සුරැකුණා.")

import os
import chromadb
import pymupdf4llm  # 🚀 Layout & Table හඳුනාගන්නා අලුත් ලයිබ්‍රරි එක
from langchain_text_splitters import MarkdownTextSplitter # Markdown වලටම විශේෂිත Splitter එකක්

# 1. Local ChromaDB එක සෙටප් කිරීම
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="financial_docs")

# Markdown ව්‍යුහය (Tables/Headers) ආරක්ෂා වන පරිදි කෑලි කඩන Splitter එක
text_splitter = MarkdownTextSplitter(chunk_size=1500, chunk_overlap=300)

pdf_folder = "./my_pdfs"
if not os.path.exists(pdf_folder):
    os.makedirs(pdf_folder)
    print(f"📁 '{pdf_folder}' folder created. Drop your PDFs inside and re-run.")
    exit()

pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith('.pdf')]

if not pdf_files:
    print("⚠️ './my_pdfs' ෆෝල්ඩර් එක ඇතුළේ PDF නැහැ!")
    exit()

id_counter = 0
for pdf in pdf_files:
    pdf_path = os.path.join(pdf_folder, pdf)
    print(f"🧠 Layout Analysis & Extracting: {pdf}")
    
    try:
        # 🚀 මුළු PDF එකම වගු ද සහිතව පිරිසිදු Markdown එකක් බවට පත් කරයි!
        md_text = pymupdf4llm.to_markdown(pdf_path)
        
        # Markdown කැබලිවලට වෙන් කිරීම
        chunks = text_splitter.split_text(md_text)
        
        for chunk in chunks:
            collection.add(
                documents=[chunk],
                metadatas=[{"source": pdf}],
                ids=[f"id_{id_counter}"]
            )
            id_counter += 1
            
    except Exception as e:
        print(f"❌ Error processing {pdf}: {e}")

print(f"\n✅ සාර්ථකයි! පිරිසිදු වගු (Markdown Tables) ද සහිතව කෑලි {id_counter}ක් සේව් වුණා.")