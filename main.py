import streamlit as st
from PIL import Image
import os
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt



header = st.container()
model_inference = st.container()
col1, col2 = st.columns(2)
features = st.container()





with header:
    # Add custom CSS styling
    st.markdown(
        """
        <style>
        .css-1aumxhk {
            background-color: red;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Render the navbar
    st.markdown(
        """
        <div class="css-1aumxhk">
        <h1 style="color: white;">COVID19 detection from Chest X-ray</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

with model_inference:
    
    st.markdown(' #### Use ML model to analyze X-ray images to detect  COVID-19 vs Normal')
    
    preprocess = transforms.Compose([
        transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load the model's parameters, mapping them to the CPU if necessary
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'efficient_b2.pth')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = models.efficientnet_b2(pretrained=True)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    def perform_inference(image):
        # Apply the transformations to the image
        input_tensor = preprocess(image)
        input_batch = input_tensor.unsqueeze(0)
        
        # Perform inference on the image
        with torch.no_grad():
            input_batch = input_batch.to(device)
            output = model(input_batch)
        
        # Get the predicted class
        _, predicted_idx = torch.max(output, 1)
        predicted_label = predicted_idx.item()
        
        # Define the class labels
        classes = ['COVID-19', 'Normal']
        
        # Get the predicted label and probabilities
        predicted_class = classes[predicted_label]
        probabilities = torch.softmax(output, dim=1)
        prob_covid = probabilities[0][0].item()
        prob_normal = probabilities[0][1].item()
        
        return predicted_class, prob_covid, prob_normal

    def upload_image():
        # Upload and display the image
        uploaded_image = st.file_uploader("Upload an image (chest X-ray image only)", type=["jpg", "jpeg", "png"])
        if uploaded_image is not None:
            image = Image.open(uploaded_image)
            # Convert grayscale image to RGB
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Perform inference on the image
            predicted_class, prob_covid, prob_normal = perform_inference(image)

            # Get image dimensions
            width, height = image.size

            # Display the image and inference results
            st.image(image, caption="Uploaded Image", width=300)
            # Convert probabilities to percentages
            prob_covid_percent = prob_covid * 100
            prob_normal_percent = prob_normal * 100

            # Plot the probabilities
            labels = ['COVID-19', 'Normal']
            probabilities = [prob_covid_percent, prob_normal_percent]
            colors = ['red', 'blue']

            fig, ax = plt.subplots()
            ax.barh(labels, probabilities, color=colors)
            ax.set_xlim(0, 100)  # Set x-axis limit from 0 to 100 (percentage range)
            ax.set_xlabel('Probability (%)')

            # Display the number values on the plot
            for i, v in enumerate(probabilities):
                ax.text(v + 1, i, str(round(v, 2)), color='black', va='center')

            # Display the image and the plot
            #st.image(image, caption="Uploaded Image", width=300)
            st.pyplot(fig)

    # Call the function to run the Streamlit app
    upload_image()


with col1:
    st.markdown("<hr>", unsafe_allow_html=True)
    st.image('static/covid19.png', caption='COVID-19', use_column_width=True)

with col2:
    st.markdown("<hr>", unsafe_allow_html=True)
    st.image('static/normal.jpg', caption='Normal', use_column_width=True)


with features:
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("### COVID-19 Detection from Chest X-ray Medical Image")
    st.markdown("""Model inference on COVID-19 chest X-rays involves using a trained machine learning model to analyze and classify X-ray images to determine whether they indicate signs of COVID-19 infection. By leveraging deep learning techniques, these models can assist healthcare professionals in diagnosing COVID-19 cases more accurately and efficiently. However, it's important to note that model predictions should be considered as an aid to medical professionals and not as a substitute for clinical diagnosis, as X-ray images alone may not provide a definitive diagnosis for COVID-19. Proper medical expertise and further diagnostic tests should always be sought for accurate diagnosis and patient care.""")
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("### Salient features for COVID-19 detection")
    st.markdown("COVID-19 chest X-ray features: Ground-glass opacities, consolidation, bilateral involvement, peripheral distribution, crazy paving pattern. These features can also be present in respiratory conditions other than COVID-19. Confirming COVID-19 requires additional tests and evaluation by healthcare professionals.")
    



