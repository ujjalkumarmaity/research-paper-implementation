from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset
import requests
from pdf2image import convert_from_path
from pypdf import PdfReader
from io import BytesIO

class M3DocragDataSet(object):
    def __init__(self,dataset_name:str = 'sample'):
        self.dataset_name = dataset_name

    def download_pdf(self,url):
        response = requests.get(url)
        if response.status_code == 200:
            return BytesIO(response.content)
        else:
            raise Exception(f"Failed to download PDF: Status code {response.status_code}")

    def get_pdf_images(self, pdf_url):
        """
        install poppler-utils, using `apt-get install poppler-utils` command
        """
        # Download the PDF
        pdf_file = self.download_pdf(pdf_url)
        # Save the PDF temporarily to disk (pdf2image requires a file path)
        temp_file = "temp.pdf"
        with open(temp_file, "wb") as f:
            f.write(pdf_file.read())
        reader = PdfReader(temp_file)
        page_texts = []
        for page_number in range(len(reader.pages)):
            page = reader.pages[page_number]
            text = page.extract_text()
            page_texts.append(text)
        images = convert_from_path(temp_file)
        assert len(images) == len(page_texts)
        return (images, page_texts)

    def load_sample_data(self):
        sample_pdfs = [
                {
                    'doc_id' : 1,
                    "url": "https://arxiv.org/pdf/1706.03762"
                }
        ]
        data = []
        for pdf in sample_pdfs:
            images, page_texts = self.get_pdf_images(pdf.get('url'))
            for page_no, (image, text) in enumerate(zip(images,page_texts)):
                data.append({
                    'doc_id' : pdf.get('doc_id'),
                    'document_page_no' : page_no,
                    'image' : images[page_no]
                })
            # pdf['images'] = images
            # pdf['texts'] = page_texts
        return Dataset.from_list(data)

        
    def load_docvaq_data(self):
        """
        Dataset url - https://www.docvqa.org/datasets

        "questionId": 52212, A unique ID number for the question
        "question": "Whose signature is given?", The question string - natural language asked question
        "image": "documents/txpn0095_1.png", The image filename corresponding to the document page which the question is defined on. The images are provided in the /documents folder
        "docId": 1968, A unique ID number for the document
        "ucsf_document_id": "txpn0095", The UCSF document id number
        "ucsf_document_page_no": "1", The page number within the UCSF document that is used here
        "answers": ["Edward R. Shannon", "Edward Shannon"], A list of correct answers provided by annotators
        "data_split": "train" The dataset split this question pertains to
        """
        data = load_dataset('lmms-lab/DocVQA','DocVQA',split='validation')
        data = data.select_columns({ 'image', 'docId','ucsf_document_page_no',}).rename_column('ucsf_document_page_no','document_page_no').rename_column('docId','doc_id')
        return data


    def load_data(self)->Dataset:
        if self.dataset_name=='sample':
            data = self.load_sample_data()
        else:
            data = self.load_docvaq_data()
        return data
    