# for the depth module MiDaS
import torch
import yaml


class DepthAnalyzer:
    def __init__(self, logger) -> None:
        # Load model settings from yaml
        with open('settings.yaml', 'r') as input_file:
            self.settings = yaml.load(input_file, yaml.FullLoader)["depth_model"]
        self.model_type = self.settings["model_type"]

        # Save the logger
        self.logger = logger

        # Load the MiDaS model
        self.midas = torch.hub.load("intel-isl/MiDaS", self.model_type)
        self.device = torch.device(
            "cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.midas.to(self.device)
        self.midas.eval()
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

        if self.model_type == "DPT_Large" or self.model_type == "DPT_Hybrid":
            self.transform = midas_transforms.dpt_transform
        else:
            self.transform = midas_transforms.small_transform

    def get_model_type(self) -> str:
        return self.model_type

    def calc_depth(self, frame, img_name) -> None:
        input_batch = self.transform(frame).to(self.device)

        # Predict and resize to original resolution
        with torch.no_grad():
            prediction = self.midas(input_batch)

            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=frame.shape[:2],
                mode=self.settings["prediction_mode"],
                align_corners=True,
            ).squeeze()

        output = prediction.cpu().numpy()

        # Save the output image using the logger
        return self.logger.save_img(output, "DEPTH", img_name)
