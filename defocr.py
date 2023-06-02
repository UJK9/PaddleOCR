import cv2
from paddleocr import PaddleOCR

def extract_cv_info(image_path):
    # Specify the path to the font file
    font_path = 'simfang.ttf'

    ocr = PaddleOCR(lang='en', det=False, rec=True, use_space_char=True, space_char_width=10, drop_score=0.5,
                    text_detector=None, text_recognizer=None, use_angle_cls=False, gpu_mem=500, use_tensorrt=False,
                    precision='fp32')

    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    # Perform OCR
    result = ocr.ocr(image)

    # Initialize dictionaries to store information
    blocks = {
        'PROFILE': [],
        'EDUCATION': [],
        'SKILLS': [],
        'EXTRACURRICULARS': []
    }

    # Extract relevant info from OCR results
    current_block = None
    for entry in result[0]:
        coordinates = entry[0]
        text = entry[1][0]

        # Logic to categorize the extracted text into different blocks
        if 'profile' in text.lower() or 'statement' in text.lower():
            current_block = 'PROFILE'
        elif 'education' in text.lower():
            current_block = 'EDUCATION'
        elif 'skills' in text.lower():
            current_block = 'SKILLS'
        elif 'extracurriculars' in text.lower():
            current_block = 'EXTRACURRICULARS'

        # Append the text to the corresponding block
        if current_block:
            blocks[current_block].append(text)

    # Save the information to a text file
    output_file = 'cv_info2.txt'
    with open(output_file, 'w') as file:
        for block, entries in blocks.items():
            file.write(block + ':\n')
            file.write('\n'.join(entries) + '\n\n')

    print("CV info saved to", output_file)
if __name__ =='__main__':

#usage
    image_path = 'cv.jpg'
    extract_cv_info(image_path)
