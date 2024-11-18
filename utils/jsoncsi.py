import json
import numpy as np
import os

class JsonCSI:
    def save_primary_csi_data(self, csi_data, file_name):
        tx = csi_data.shape[1]
        rx = csi_data.shape[2]
        for t in range(0, tx):
            for r in range(0, rx):
                data = csi_data[:,t,r,:]
                data_list = {
                    'csi': [[[val.real, val.imag] for val in row] for row in data]
                }

                path = f'json data/{file_name}'
                json_file_name = f'tx-{t+1} rx-{r+1}.json'
                if not os.path.exists(path):
                    os.makedirs(path)

                with open(os.path.join(path, json_file_name), 'w') as f:
                    json.dump(data_list, f)

    def load_csi_data(self, csi_name, tx, rx, smoothed):
        if smoothed:
            file_path = f'{csi_name}/tx-{tx} rx-{rx} smoothed.json'
        else:
            file_path = f'{csi_name}/tx-{tx} rx-{rx}.json'
        full_path = os.path.join('json data', file_path)

        if os.path.exists(full_path):
            with open(full_path, 'r') as f:
                data = json.load(f)
            if smoothed:
                csi_data_reconstructed = np.array(data['csi_smoothed'])
            else:
                csi_data_reconstructed = np.array([[complex(real, imag) for real, imag in row] for row in data['csi']])
            return csi_data_reconstructed
        return None

    def save_smoothed_csi_data(self, csi_smoothed_data, file_name, tx, rx):
        data_list = {
            'csi_smoothed': csi_smoothed_data.tolist()
        }

        path = f'json data/{file_name}'
        json_file_name = f'tx-{tx} rx-{rx} smoothed.json'
        if not os.path.exists(path):
            os.makedirs(path)

        with open(os.path.join(path, json_file_name), 'w') as f:
            json.dump(data_list, f)
