import tempfile
from fpdf import FPDF

def create_pdf_report(report_text: str, summary_fig, dep_fig1, dep_fig2) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Process text report
    for line in report_text.split('\n'):
        line = line.encode('latin-1', 'replace').decode('latin-1').replace('**', '').strip()
        
        if not line:
            pdf.ln(5)
            continue
            
        if line.startswith('# '):
            pdf.set_font("Helvetica", 'B', 16)
            pdf.multi_cell(w=0, h=10, txt=line.replace('# ', ''), new_x="LMARGIN", new_y="NEXT")
        elif line.startswith('## '):
            pdf.set_font("Helvetica", 'B', 14)
            pdf.multi_cell(w=0, h=8, txt=line.replace('## ', ''), new_x="LMARGIN", new_y="NEXT")
        elif line.startswith('### '):
            pdf.set_font("Helvetica", 'B', 12)
            pdf.multi_cell(w=0, h=8, txt=line.replace('### ', ''), new_x="LMARGIN", new_y="NEXT")
        else:
            pdf.set_font("Helvetica", '', 12)
            pdf.multi_cell(w=0, h=6, txt=line, new_x="LMARGIN", new_y="NEXT")
            
    # Save figures to tempfiles and add to pdf
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f1, \
         tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f2, \
         tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f3:
        
        # Adding some spacing before figures
        pdf.add_page()
        summary_fig.savefig(f1.name, format="png", bbox_inches="tight")
        dep_fig1.savefig(f2.name, format="png", bbox_inches="tight")
        dep_fig2.savefig(f3.name, format="png", bbox_inches="tight")
        
        pdf.set_font("Helvetica", 'B', 14)
        pdf.multi_cell(w=0, h=10, txt="Feature Importance (SHAP Summary)", new_x="LMARGIN", new_y="NEXT")
        pdf.image(f1.name, w=170)
        
        pdf.add_page()
        pdf.multi_cell(w=0, h=10, txt="Median Income Dependence", new_x="LMARGIN", new_y="NEXT")
        pdf.image(f2.name, w=170)
        
        pdf.add_page()
        pdf.multi_cell(w=0, h=10, txt="Average Occupancy Dependence", new_x="LMARGIN", new_y="NEXT")
        pdf.image(f3.name, w=170)
        
    return bytes(pdf.output())
