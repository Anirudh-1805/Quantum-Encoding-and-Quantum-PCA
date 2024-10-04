from flask import Flask, render_template, request, send_file, make_response
import pandas as pd
import io
import zipfile
from amplitude_encoding import amplitude_encoding  # Import the function
from angle_encoding import apply_rx_encoding, apply_ry_encoding, apply_rz_encoding

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/process', methods=['POST'])
def process_file():
    file = request.files['file']
    operations = request.form.getlist('operations')

    # Convert the uploaded file to a DataFrame
    df = pd.read_csv(file)
    df2 = df.copy()

    # Perform the selected operations
    if 'Amplitude Encoding' in operations:
        df = amplitude_encoding(df.copy())  # Apply amplitude encoding

        output_amp = io.StringIO()
        df.to_csv(output_amp, index=False)
        output_amp.seek(0)

        response_amp = make_response(send_file(io.BytesIO(output_amp.getvalue().encode()),
                                               mimetype='text/csv',
                                               as_attachment=True,
                                               download_name='amp_encoded.csv'))
        
    
    if 'Angle Encoding' in operations:
        angle_option = request.form.get('angle_option')
        if angle_option:
            # Perform the selected angle encoding
            if angle_option == 'Rx':
                df_angle_encoded = apply_rx_encoding(df2.copy())
            elif angle_option == 'Ry':
                df_angle_encoded = apply_ry_encoding(df2.copy())
            elif angle_option == 'Rz':
                df_angle_encoded = apply_rz_encoding(df2.copy())
            
            # Prepare the angle encoded file for download
            output_angle = io.StringIO()
            df_angle_encoded.to_csv(output_angle, index=False)
            output_angle.seek(0)
            
            response_angle = make_response(send_file(io.BytesIO(output_angle.getvalue().encode()),
                                                     mimetype='text/csv',
                                                     as_attachment=True,
                                                     download_name='ang_encoded.csv'))
            
    
    if 'Amplitude Encoding' in operations and 'Angle Encoding' in operations:
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            zip_file.writestr('amp_encoded.csv', output_amp.getvalue())
            zip_file.writestr('ang_encoded.csv', output_angle.getvalue())
        
        zip_buffer.seek(0)

        return send_file(zip_buffer, mimetype='application/zip', as_attachment=True, download_name='encoded_files.zip')
    
    # Return the appropriate response based on selected operation
    elif 'Amplitude Encoding' in operations:
        return response_amp
    elif 'Angle Encoding' in operations:
        return response_angle
    else:
        return "No encoding selected", 400
    

if __name__ == '__main__':
    app.run(debug=True)
    