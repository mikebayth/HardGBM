#include "rain/weights.h"
#include "activations.h"
#include <hls_video.h>

#include "ap_fixed.h"

// #define EXP_WIDTH	4
// #define INT_WIDTH	2

// typedef ap_fixed<EXP_WIDTH, INT_WIDTH> float24_t;


void fc_layer1(hls::stream<float24_t> &out, hls::stream<float24_t> &in,
		float24_t weight[FC1_WEIGHTS_H][FC1_WEIGHTS_W],
		float24_t bias[FC1_BIAS_SIZE]) {
	float24_t read;
	float24_t output[FC1_ACT_SIZE] = { 0 };

	in >> read;
	for (int i = 0; i < FC1_WEIGHTS_W; i++)
#pragma HLS PIPELINE II=1
		output[i] = weight[0][i] * read;

	fc_layer1_label12: for (int j = 1; j < FC1_WEIGHTS_H; j++) {
		in >> read;
		fc_layer1_label40: for (int i = 0; i < FC1_WEIGHTS_W; i++) {
			output[i] += weight[j][i] * read;
		}
	}
	fc_layer1_label15: for (int i = 0; i < FC1_WEIGHTS_W; i++)
#pragma HLS PIPELINE II=1
		out << relu(output[i] + bias[i]);

}

void fc_layer2(hls::stream<float24_t> &out, hls::stream<float24_t> &in,
		float24_t weight[FC2_WEIGHTS_H][FC2_WEIGHTS_W],
		float24_t bias[FC2_BIAS_SIZE]) {
	float24_t read;
	float24_t output[FC2_ACT_SIZE] = { 0 };

	in >> read;
	for (int i = 0; i < FC2_WEIGHTS_W; i++)
#pragma HLS PIPELINE II=1
		output[i] = weight[0][i] * read;

	fc_layer2_label13: for (int j = 1; j < FC2_WEIGHTS_H; j++) {
		in >> read;
		fc_layer2_label41: for (int i = 0; i < FC2_WEIGHTS_W; i++) {
			output[i] += weight[j][i] * read;
		}
	}
	fc_layer2_label11: for (int i = 0; i < FC2_WEIGHTS_W; i++)
#pragma HLS PIPELINE II=1
		out << relu(output[i] + bias[i]);

}

void fc_layer3(hls::stream<float24_t> &out, hls::stream<float24_t> &in,
		float24_t weight[FC3_WEIGHTS_H][FC3_WEIGHTS_W],
		float24_t bias[FC3_BIAS_SIZE]) {
	float24_t read;
	float24_t output[FC3_ACT_SIZE] = { 0 };

	in >> read;
	for (int i = 0; i < FC3_WEIGHTS_W; i++)
#pragma HLS PIPELINE II=1
		output[i] = weight[0][i] * read;

	fc_layer3_label10: for (int j = 1; j < FC3_WEIGHTS_H; j++) {
		in >> read;
		fc_layer3_label42: for (int i = 0; i < FC3_WEIGHTS_W; i++) {
			output[i] += weight[j][i] * read;
		}
	}
	fc_layer3_label14: for (int i = 0; i < FC3_WEIGHTS_W; i++)
#pragma HLS PIPELINE II=1
		out << relu(output[i] + bias[i]);

}

void fc_layer4(hls::stream<float24_t> &out, hls::stream<float24_t> &in,
		float24_t weight[FC4_WEIGHTS_H][FC4_WEIGHTS_W],
		float24_t bias[FC4_BIAS_SIZE]) {
	float24_t read;
	float24_t output[FC4_ACT_SIZE] = { 0 };

	in >> read;
	for (int i = 0; i < FC4_WEIGHTS_W; i++)
#pragma HLS PIPELINE II=1
		output[i] = weight[0][i] * read;

	fc_layer4_label10: for (int j = 1; j < FC4_WEIGHTS_H; j++) {
		in >> read;
		fc_layer4_label42: for (int i = 0; i < FC4_WEIGHTS_W; i++) {
			output[i] += weight[j][i] * read;
		}
	}
	fc_layer4_label14: for (int i = 0; i < FC4_WEIGHTS_W; i++)
#pragma HLS PIPELINE II=1
		out << relu(output[i] + bias[i]);

}

void nnet(hls::stream<float24_t> &fc4_out) {

	hls::stream<float24_t> fc1_out("fc1_out");
	hls::stream<float24_t> fc2_out("fc2_out");
	hls::stream<float24_t> fc3_out("fc3_out");
	hls::stream<float24_t> input("input");

	// for (int i = 0; i < IMAGE_SIZE * IMAGE_SIZE * IMAGE_CHANNELS; i++)
	// 	image_in << image[i];
	for (int i = 0; i <FC1_WEIGHTS_H ; i++)
	input << input_vector[i];


	fc_layer1(fc1_out, input, fc1_weight, fc1_bias);
	fc_layer2(fc2_out, fc1_out, fc2_weight, fc2_bias);
	fc_layer3(fc3_out, fc2_out, fc3_weight, fc3_bias);
	fc_layer4(fc4_out, fc3_out, fc4_weight, fc4_bias);
}
