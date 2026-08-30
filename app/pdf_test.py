import fitz
import os

pdf_path = "data/raw/sample.pdf"

text_output = "data/processed/sample_text.txt"
image_output_dir = "data/processed/images"

os.makedirs(os.path.dirname(text_output), exist_ok=True)
os.makedirs(image_output_dir, exist_ok=True)

document = fitz.open(pdf_path)

with open(text_output, "w", encoding="utf-8") as output_file:

    for page_number, page in enumerate(document):

        # -------- TEXT EXTRACTION --------
        text = page.get_text()

        output_file.write(
            f"\n\n===== PAGE {page_number + 1} =====\n\n"
        )

        output_file.write(text)

        # -------- IMAGE EXTRACTION --------
        images = page.get_images(full=True)

        for image_index, image in enumerate(images):

            xref = image[0]

            image_data = document.extract_image(xref)

            image_bytes = image_data["image"]
            image_ext = image_data["ext"]

            image_filename = (
                f"page_{page_number + 1}_image_{image_index + 1}.{image_ext}"
            )

            image_path = os.path.join(
                image_output_dir,
                image_filename
            )

            with open(image_path, "wb") as image_file:
                image_file.write(image_bytes)

            print("Extracted:", image_path)

document.close()

print("\nPDF extraction completed!")
print("Text saved to:", text_output)
print("Images saved to:", image_output_dir)