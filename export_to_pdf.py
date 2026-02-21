import markdown
from xhtml2pdf import pisa
import os

def convert_md_to_pdf(md_file, pdf_file):
    # Read Markdown file
    with open(md_file, 'r', encoding='utf-8') as f:
        md_content = f.read()

    # Convert Markdown to HTML
    # We add some CSS for a professional look
    html_content = markdown.markdown(md_content, extensions=['tables'])
    
    # Custom CSS for the PDF layout
    css = """
    <style>
        @page {
            size: a4 portrait;
            @frame content_frame {
                left: 2cm; width: 17cm; top: 2cm; height: 25cm;
            }
        }
        body {
            font-family: Arial, sans-serif;
            font-size: 11pt;
            color: #333;
            line-height: 1.5;
        }
        h1 { color: #1a5276; text-align: center; border-bottom: 2px solid #1a5276; padding-bottom: 10px; }
        h2 { color: #1a5276; border-bottom: 1px solid #ccc; padding-top: 15px; margin-bottom: 5px; }
        h3 { color: #2e86c1; margin-top: 15px; }
        table { width: 100%; border-collapse: collapse; margin-top: 10px; }
        th, td { border: 1px solid #ccc; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; color: #1a5276; }
        img { display: block; margin: 20px auto; width: 450px; }
        .footer { text-align: center; font-style: italic; font-size: 9pt; color: #777; margin-top: 50px; }
    </style>
    """
    
    full_html = f"<html><head>{css}</head><body>{html_content}</body></html>"

    # Convert HTML to PDF
    with open(pdf_file, "wb") as f_pdf:
        pisa_status = pisa.CreatePDF(full_html, dest=f_pdf)

    return not pisa_status.err

if __name__ == "__main__":
    md_path = "Report_Churn_Analysis.md"
    pdf_path = "Rapport_Final_Churn.pdf"
    
    print(f"🔄 Conversion de {md_path} en PDF...")
    if convert_md_to_pdf(md_path, pdf_path):
        print(f"✅ Succès ! Le rapport PDF est disponible ici : {os.path.abspath(pdf_path)}")
    else:
        print("❌ Erreur lors de la création du PDF.")
