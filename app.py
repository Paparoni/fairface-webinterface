import streamlit as st
import os
import hashlib
import pandas as pd
import subprocess
import matplotlib.pyplot as plt
import numgen
st.set_page_config(page_title='Facial Recoginition', layout="wide")

# Add sidebar
st.sidebar.title("Navigation")
nav = st.sidebar.radio("Go to", ["Home", "About"])

# Handle sidebar selection
if nav == "Home":
    # Define the path to the test folder
    test_folder = os.path.join(os.getcwd(), 'test')

    # Define the path to the test CSV file
    test_csv_file = os.path.join(os.getcwd(), 'test_imgs.csv')

    # Create the test folder if it doesn't already exist
    if not os.path.exists(test_folder):
        os.makedirs(test_folder)

    # Create the test CSV file if it doesn't already exist
    if not os.path.exists(test_csv_file):
        df = pd.DataFrame(columns=['img_path'])
        df.to_csv(test_csv_file, index=False)

    # Create a Streamlit app
    def app():
        # Set the page title
        st.title('Facial Recoginition')
        st.write('This app uses the fairface model to accurately produce preditions on race, gender, and age on detected faces. This app allows the user to upload images and then be returned a dataframe with the predictions.')
        
        # Create a file uploader
        uploaded_files = st.file_uploader('Upload one or more images',
                                        type=['jpg', 'jpeg', 'png'],
                                        accept_multiple_files=True)
        
        if st.button('Upload'):
            # If the user uploads an image, save it to the test folder
            if uploaded_files is not None:
                highlighted_rows = []
                for uploaded_file in uploaded_files:
                    # Generate a unique name for the uploaded image using the current time and the SHA-256 hash value of the image cryptographically secure
                    rand_num_str = str(numgen.generate())
                    file_data = uploaded_file.getvalue()
                    file_hash = hashlib.sha256(file_data).hexdigest()[:10]
                    file_name = file_hash + '_' + rand_num_str + '.jpg'
                    file_path = os.path.join(test_folder, file_name)
                    
                    # Save the uploaded image to the test folder
                    with open(file_path, 'wb') as f:
                        f.write(file_data)
                    st.success(f'Successfully saved {file_name} to {test_folder}')
                    
                    # Add the image path to the test CSV file
                    df = pd.read_csv(test_csv_file)
                    df = df.append({'img_path': 'test/' + file_name}, ignore_index=True)
                    df.to_csv(test_csv_file, index=False)
                    st.success(f'Successfully added {file_name} to {test_csv_file}')
                    
                    # Add the filename of the uploaded image to the list of highlighted rows
                    highlighted_rows.append(file_name)
                
                # Run the predict.py script
                # Use a spinner to indicate the script running
                st.write('Prediction')
                with st.spinner('Running prediction script...'):
                    cmd = f'python predict.py --csv {test_csv_file}'
                    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    stdout, stderr = process.communicate()
                    return_code = process.poll()
                    if return_code != 0:
                        st.error(stderr.decode())
                    else:
                        st.success(stdout.decode())
                        post_process(highlighted_rows)

    def post_process(highlighted_rows):
        # Load the data
        data = pd.read_csv('test_outputs.csv')

        highlighted_rows = list(map(lambda x: x.replace('.jpg', ''), highlighted_rows))

        # Create a function that highlights the rows
        def highlight_rows(row):
            background = 'yellow' if any(s in row['face_name_align'] for s in highlighted_rows) else '#0e1117'
            return ['background-color: %s' % background]*len(row)

        # Apply the function to each row of the DataFrame
        styled_data = data.style.apply(highlight_rows, axis=1)

        # Display the styled DataFrame
        st.write(styled_data)


    app()

elif nav == "About":
    st.title("About Page")
    # Add information the app here
