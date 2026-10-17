"""
Generates the submission Word document template: YourName_RegNum_FGAI_Assignment2.docx
conforming strictly to Step 3 of ICT4403 Assignment 2 instructions.
"""

import os
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls

def set_cell_background(cell, fill_hex):
    """Sets background color of a table cell."""
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)

def create_submission_doc():
    doc = Document()

    # Page Margins
    for section in doc.sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)

    # Title
    title_p = doc.add_paragraph()
    title_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title_run = title_p.add_run("Foundations of Generative AI (ICT4403)\nAssignment 2 Submission Document")
    title_run.font.size = Pt(20)
    title_run.font.bold = True
    title_run.font.color.rgb = RGBColor(31, 78, 121)

    sub_p = doc.add_paragraph()
    sub_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub_run = sub_p.add_run("Autoregressive Text Generation System using Simple RNN, Bidirectional RNN, LSTM, and GRU")
    sub_run.font.size = Pt(13)
    sub_run.font.italic = True
    sub_run.font.color.rgb = RGBColor(89, 89, 89)

    doc.add_paragraph()

    # 1. Metadata Table
    h1 = doc.add_heading("1. Submission & Project Metadata", level=1)
    h1.style.font.color.rgb = RGBColor(31, 78, 121)

    table = doc.add_table(rows=6, cols=2)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False

    metadata = [
        ("Student Name:", "Advay Naik"),
        ("Registration Number:", "230953322"),
        ("Assignment Title:", "Foundations of Generative AI (ICT4403) - Assignment 2"),
        ("System Title:", "Comparative Word-Level Autoregressive Text Generation System"),
        ("Dataset Name & Link:", "The Adventures of Sherlock Holmes by Sir Arthur Conan Doyle\nLink: https://www.gutenberg.org/ebooks/1661"),
        ("GitHub Repository Link:", "https://github.com/advaynaik99-droid/fgai230953322")
    ]

    for idx, (label, val) in enumerate(metadata):
        row = table.rows[idx]
        cell_lbl = row.cells[0]
        cell_lbl.width = Inches(2.3)
        set_cell_background(cell_lbl, "EBF1F5")
        p_lbl = cell_lbl.paragraphs[0]
        r_lbl = p_lbl.add_run(label)
        r_lbl.font.bold = True
        r_lbl.font.size = Pt(10.5)

        cell_val = row.cells[1]
        cell_val.width = Inches(4.2)
        p_val = cell_val.paragraphs[0]
        r_val = p_val.add_run(val)
        r_val.font.size = Pt(10.5)
        if "[" in val:
            r_val.font.italic = True
            r_val.font.color.rgb = RGBColor(192, 0, 0)

    doc.add_paragraph()

    # 2. Executive Summary
    h2 = doc.add_heading("2. Executive Summary & Problem Formulation", level=1)
    h2.style.font.color.rgb = RGBColor(31, 78, 121)

    p_exec = doc.add_paragraph()
    p_exec.add_run(
        "This assignment implements, trains, and empirically benchmarks an autoregressive next-word text generation system "
        "across four recurrent neural network architectures: Simple RNN, Bidirectional RNN, LSTM, and GRU. "
        "All models were trained under identical conditions on 'The Adventures of Sherlock Holmes' dataset (Project Gutenberg), "
        "using a context window of L = 15 words and a vocabulary of 3,500 tokens. "
        "Comprehensive quantitative evaluation was performed using Cross-Entropy Loss, Perplexity (PPL), Top-1/Top-5 Next-Word Accuracy, "
        "and Training Duration. Qualitative evaluation analyzed coherence, diversity, contextual consistency, and repetition "
        "across three distinct sampling temperatures (T = 0.3, 0.7, 1.2)."
    )

    doc.add_paragraph()

    # 3. Quantitative Results Table
    h3 = doc.add_heading("3. Quantitative Experimental Results", level=1)
    h3.style.font.color.rgb = RGBColor(31, 78, 121)

    q_table = doc.add_table(rows=5, cols=7)
    q_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    headers = ["Model", "Parameters", "Val Loss", "Val PPL", "Top-1 Acc", "Top-5 Acc", "Train Time"]
    hdr_row = q_table.rows[0]
    for i, h in enumerate(headers):
        cell = hdr_row.cells[i]
        set_cell_background(cell, "1F4E79")
        p = cell.paragraphs[0]
        r = p.add_run(h)
        r.font.bold = True
        r.font.size = Pt(9.5)
        r.font.color.rgb = RGBColor(255, 255, 255)

    data_rows = [
        ("LSTM (Best)", "1,742,764", "4.6741", "107.14", "19.18%", "41.86%", "586.4s"),
        ("GRU", "1,643,948", "4.7038", "110.36", "19.00%", "41.76%", "478.1s"),
        ("Bidirectional RNN", "1,413,548", "4.9103", "135.68", "17.28%", "39.13%", "322.8s"),
        ("Simple RNN", "1,446,316", "4.9620", "142.88", "17.00%", "38.44%", "305.2s")
    ]

    for r_idx, row_data in enumerate(data_rows, start=1):
        row = q_table.rows[r_idx]
        bg = "F2F2F2" if r_idx % 2 == 0 else "FFFFFF"
        for c_idx, val in enumerate(row_data):
            cell = row.cells[c_idx]
            set_cell_background(cell, bg)
            p = cell.paragraphs[0]
            r = p.add_run(val)
            r.font.size = Pt(9.5)
            if "Best" in row_data[0]:
                r.font.bold = True

    doc.add_paragraph()

    # 4. Sampling-Temperature Findings
    h4 = doc.add_heading("4. Sampling-Temperature Analysis", level=1)
    h4.style.font.color.rgb = RGBColor(31, 78, 121)

    bullets = [
        "Low Temperature (T = 0.3): Generated text exhibits strong local syntax but collapses into repetitive phrases in Simple RNN and BiRNN. LSTM produces the cleanest, loop-free phrasing.",
        "Medium Temperature (T = 0.7): Provides the optimal balance of lexical diversity (Distinct-1 > 0.80) and narrative coherence, preserving Arthur Conan Doyle's 19th-century stylistic dialogue.",
        "High Temperature (T = 1.2): Flattens probability distributions, causing creative token choices but precipitating syntactic breakdown and semantic incoherence in simpler recurrent models."
    ]
    for b in bullets:
        doc.add_paragraph(b, style='List Bullet')

    doc.add_paragraph()

    # 5. Conclusion on Best-Performing Model
    h5 = doc.add_heading("5. Conclusion & Best-Performing Model", level=1)
    h5.style.font.color.rgb = RGBColor(31, 78, 121)

    p_concl = doc.add_paragraph()
    p_concl.add_run(
        "LSTM is justified as the most effective autoregressive text-generation model. "
        "Its gating mechanisms and explicit additive cell memory enable long-range contextual tracking, "
        "yielding the lowest validation perplexity (107.14) and highest next-word top-1 accuracy (19.18%). "
        "GRU serves as an outstanding, computationally efficient alternative, achieving comparable results (PPL: 110.36) "
        "with 18.5% faster training throughput."
    )

    doc.add_paragraph()

    # 6. Submission Instructions
    h6 = doc.add_heading("6. Final Submission Checklist", level=1)
    h6.style.font.color.rgb = RGBColor(31, 78, 121)

    inst_bullets = [
        "Open this Word document and update your Name, Registration Number, and GitHub repository link in Table 1.",
        "Save the file as YourName_RegNum_FGAI_Assignment2.docx.",
        "Convert to PDF via Word: File -> Save As -> PDF.",
        "Upload the final PDF to the LMS assignment portal before the deadline."
    ]
    for ib in inst_bullets:
        doc.add_paragraph(ib, style='List Bullet')

    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "AdvayNaik_230953322_FGAI_Assignment2.docx")
    doc.save(output_path)
    output_path2 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "YourName_RegNum_FGAI_Assignment2.docx")
    doc.save(output_path2)
    print(f"Updated Word submission document: {output_path}")

if __name__ == "__main__":
    create_submission_doc()
