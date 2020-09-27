import streamlit as st
from PIL import Image
from keras.models import load_model
import matplotlib.pyplot as plt
import cv2
import numpy as np
import io
import os
try:
    model=load_model('mnist.h5')
except:
    st.write("Cannot load mnist model")
#st.title("WELCOME TO MNIST DIGIT RECOGNIZATION:sunglasses:")
def my_model(image):
    st.image(image,width=600)
    image = np.array(image)
    #images = cv2.imread(image)
    imagess=cv2.imwrite('out.jpg',image)
    images=cv2.imread('out.jpg')
    blu=cv2.GaussianBlur(images,(3,3),0)
    grey = cv2.cvtColor(blu.copy(), cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(grey.copy(), 65, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    preprocessed_digits = []
    a=[]
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)

        # Creating a rectangle around the digit in the original image (for displaying the digits fetched via contours)
        cv2.rectangle(image, (x,y), (x+w, y+h), color=(0, 255, 0), thickness=2)

        # Cropping out the digit from the image corresponding to the current contours in the for loop
        digit = thresh[y:y+h, x:x+w]

        # Resizing that digit to (18, 18)
        resized_digit = cv2.resize(digit, (18,18))

        # Padding the digit with 5 pixels of black color (zeros) in each side to finally produce the image of (28, 28)
        padded_digit = np.pad(resized_digit, ((5,5),(5,5)), "constant", constant_values=0)

        # Adding the preprocessed digit to the list of preprocessed digits
        preprocessed_digits.append(padded_digit)
    #print("\n\n\n----------------Contoured Image--------------------")
    #plt.imshow(image, cmap="gray")
    #plt.show()
    inp = np.array(preprocessed_digits)
    for digit in preprocessed_digits:
        prediction = model.predict(digit.reshape(1, 28, 28, 1))  
        print('output is ',np.argmax(prediction))
        a.append(np.argmax(prediction))
    a.reverse()
    nums=len(a)
    return image,a,nums
def about():
    st.write("Welcome to my page\nHere I had implemented digit Recognization using Keras Library and dataset used is MNIST")
    st.write("The MNIST database is a large database of handwritten digits that is commonly used for training various image processing systems. The database is also widely used for training and testing in the field of machine learning. It was created by remixing the samples from NISTs original datasets")
def main():
    name=st.text_input("**Enter your name**")
    z=""
    #st.write(name)
    if name is not z:
        name1=str.upper(name)
        st.title("Hello"+" "+str(name1)+" "+"WELCOME TO MNIST DIGIT RECOGNIZATION:sunglasses:")
        st.write("**By S.Sivasai**")
        activities = ["Home", "About"]
        choice = st.sidebar.selectbox("Pick something fun", activities)
        if choice == "Home":
            st.write("Go to the About section from the sidebar to learn more about it.")
            # You can specify more file types below if you want
            #image_file = st.file_uploader("Upload image", type=['jpeg', 'png', 'jpg', 'webp'])
            st.write("Please follow the below instructions:")
            st.warning("1) While Uploading image please make ensure that numbers are written with a black sketch or marker pen\n")
            st.warning("2)picture took at bright light condition\n")
            st.warning("3)crop it to appear with only Those Numbers To get the result accurately")
            # You can specify more file types below if you want
            image_file = st.file_uploader("Upload image", type=['jpeg', 'png', 'jpg', 'webp'])
            #st.write("Input image given by you")
            #st.image(image,width=600)
            if image_file is not None:
                text_io = io.TextIOWrapper(image_file)
                st.set_option('deprecation.showfileUploaderEncoding', False)
                img = Image.open(image_file)
                if st.button("Process"):
                    res_image,a,nums=my_model(img)
                    st.write("{} numbers are detected".format(nums))
                    st.write("numbers are {}".format(a))
                    st.image(res_image,width=600)
        elif choice == "About":
            about()
    else:
        st.write("Please Enter Your name")
if __name__=="__main__":
    main()
    
                

