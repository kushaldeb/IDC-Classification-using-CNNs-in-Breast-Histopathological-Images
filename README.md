# IDC-Classification-using-CNNs-in-Breast-Histopathological-Images
Classifying between IDC (Invasive Ductal Carcinoma) and Non-IDC cells in Breast Histopathological images using various state-of-the-art Convolutional Neural Networks (CNN).

<b>Abstract</b> - Breast cancer is a disease that occurs when cells in breast tissue change (or mutate) and keep reproducing. These abnormal cells usually cluster together to form a tumour. Invasive ductal carcinoma (IDC), also known as infiltrating ductal carcinoma is the most common form of breast cancer, representing 80 percent of all breast cancer diagnoses. It is a cancer that begins growing in a milk duct and invades the fibrous or fatty tissue of the breast outside of the duct.
Globally, breast cancer is the most common cancer among women, after skin cancer. It is also the second leading cause of cancer death in women after lung cancer. Breast cancer ranks as the number one cancer among Indian females with rate as high as 25.8 per 1,00,000 women and mortality of 12.7 per 1,00,000 women according to the health ministry.</br>
In this project I will classify IDC and Non-IDC cells from Breast Histopathological Images using various state-of-the-art Convolutional Neural Network (CNN) architectures namely <b>LeNet</b>, <b>AlexNet</b>, <b>ZFNet</b>, <b>VGG16</b>, <b>VGG19</b>, <b>Inception</b>, <b>InceptionResnet</b>, <b>Densenet</b>, <b>Xception</b> and <b>Mobilenet</b>. The dataset contains 198,738 Non-IDC images and 78,786 IDC images collected from 162 whole mount slide images of Breast Cancer (BCa) specimen scanned at 40x. The breast tissues are stained with an artificial dye. In this classification system IDC cells are classified as 1 and Non-IDC cells are classified as 0. Conventional methods often take more time in analysis. This system is not meant to be a stand-alone system but a means to assist doctors and pathology technicians to classify correctly faster and better. Proper and early classification of IDC cells will result in better mortality rate among women with breast cancer.

<hr/>

<b>Project Outline</b> - All the images are inside their respective patient IDs. Python scripts move-1.py and move-3.py are used to create the training, validation and test dataset. <br/>

After training 10 CNNs on the dataset used the top 5 best performing models to create an Ensemble models to get best precision.
