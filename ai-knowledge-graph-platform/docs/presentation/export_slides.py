import win32com.client
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
pptx_path = os.path.join(SCRIPT_DIR, "GraphRAG_Pitch.pptx")
output_dir = os.path.join(SCRIPT_DIR, "slide_images")
os.makedirs(output_dir, exist_ok=True)

ppt = win32com.client.Dispatch("PowerPoint.Application")
ppt.Visible = 1
presentation = ppt.Presentations.Open(pptx_path, False, False, True)
slide_count = presentation.Slides.Count
print(f"Slide count: {slide_count}")

for i in range(1, slide_count + 1):
    out_path = os.path.join(output_dir, f"slide-{i}.jpg")
    presentation.Slides(i).Export(out_path, "JPG", 1920, 1080)
    print(f"Exported slide {i} -> {out_path}")

presentation.Close()
ppt.Quit()
print("Done.")
