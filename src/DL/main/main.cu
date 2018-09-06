#ifdef SSD
#include <stdio.h>
#include "DL/util/net.h"
#include "DL/util/layer_factory.h"
#include <cuda_runtime.h>
#include <time.h>

void confignet(struct Net* net){
	  net->layer_num = 0;
	  net->tops_num = 0;
	  net->tops = NULL;
	  net->names = NULL;
	  net->layer_parameter = NULL;

	  char* input[1] = {"data"};
	  make_input(net, "data", 1, input, 1, 3, 448, 448);

	  char* b_1[1] = {"data"};
	  char* t_1[1] = {"conv1"};
	  make_convolution(net, "conv1", 1, b_1, 1, t_1, 64, 3, 2, 7, 1);

	  char* b_2[1] = {"conv1"};
	  char* t_2[1] = {"ReLU1"};
	  make_relu(net, "ReLU1", 1, b_2, 1, t_2, 0.1);

	  char* b_3[1] = {"ReLU1"};
	  char* t_3[1] = {"pool1"};
	  make_pooling(net, "pool1", 1, b_3, 1, t_3, 0, 2, 2, MAX, 0);

	  char* b_4[1] = {"pool1"};
	  char* t_4[1] = {"conv2"};
	  make_convolution(net, "conv2", 1, b_4, 1, t_4, 192, 1, 1, 3, 1);

	  char* b_5[1] = {"conv2"};
	  char* t_5[1] = {"ReLU2"};
	  make_relu(net, "ReLU2", 1, b_5, 1, t_5, 0.1);

	  char* b_6[1] = {"ReLU2"};
	  char* t_6[1] = {"pool2"};
	  make_pooling(net, "pool2", 1, b_6, 1, t_6, 0, 2, 2, MAX, 0);

	  char* b_7[1] = {"pool2"};
	  char* t_7[1] = {"conv3"};
	  make_convolution(net, "conv3", 1, b_7, 1, t_7, 128, 0, 1, 1, 1);

	  char* b_8[1] = {"conv3"};
	  char* t_8[1] = {"ReLU3"};
	  make_relu(net, "ReLU3", 1, b_8, 1, t_8, 0.1);

	  char* b_9[1] = {"ReLU3"};
	  char* t_9[1] = {"conv4"};
	  make_convolution(net, "conv4", 1, b_9, 1, t_9, 256, 1, 1, 3, 1);

	  char* b_10[1] = {"conv4"};
	  char* t_10[1] = {"ReLU4"};
	  make_relu(net, "ReLU3", 1, b_10, 1, t_10, 0.1);

	  char* b_11[1] = {"ReLU4"};
	  char* t_11[1] = {"conv5"};
	  make_convolution(net, "conv4", 1, b_11, 1, t_11, 256, 0, 1, 1, 1);

	  char* b_12[1] = {"conv5"};
	  char* t_12[1] = {"ReLU5"};
	  make_relu(net, "ReLU5", 1, b_12, 1, t_12, 0.1);

	  char* b_13[1] = {"ReLU5"};
	  char* t_13[1] = {"conv6"};
	  make_convolution(net, "conv4", 1, b_13, 1, t_13, 512, 1, 1, 3, 1);

	  char* b_14[1] = {"conv6"};
	  char* t_14[1] = {"ReLU6"};
	  make_relu(net, "ReLU6", 1, b_14, 1, t_14, 0.1);

	  char* b_15[1] = {"ReLU6"};
	  char* t_15[1] = {"pool3"};
	  make_pooling(net, "pool3", 1, b_15, 1, t_15, 0, 2, 2, MAX, 0);

	  char* b_16[1] = {"pool3"};
	  char* t_16[1] = {"conv7"};
	  make_convolution(net, "conv7", 1, b_16, 1, t_16, 256, 0, 1, 1, 1);

	  char* b_17[1] = {"conv7"};
	  char* t_17[1] = {"ReLU7"};
	  make_relu(net, "ReLU7", 1, b_17, 1, t_17, 0.1);

	  char* b_18[1] = {"ReLU7"};
	  char* t_18[1] = {"conv8"};
	  make_convolution(net, "conv8", 1, b_18, 1, t_18, 512, 1, 1, 3, 1);

	  char* b_19[1] = {"conv8"};
	  char* t_19[1] = {"ReLU8"};
	  make_relu(net, "ReLU8", 1, b_19, 1, t_19, 0.1);

	  char* b_20[1] = {"ReLU8"};
	  char* t_20[1] = {"conv9"};
	  make_convolution(net, "conv9", 1, b_20, 1, t_20, 256, 0, 1, 1, 1);

	  char* b_21[1] = {"conv9"};
	  char* t_21[1] = {"ReLU9"};
	  make_relu(net, "ReLU9", 1, b_21, 1, t_21, 0.1);

	  char* b_22[1] ={"ReLU9"};
	  char* t_22[1] = {"conv10"};
	  make_convolution(net, "conv10", 1, b_22, 1, t_22, 512, 1, 1, 3, 1);

	  char* b_23[1] ={"conv10"};
	  char* t_23[1] = {"ReLU10"};
	  make_relu(net, "ReLU10", 1, b_23, 1, t_23, 0.1);

	  char* b_24[1] ={"ReLU10"};
	  char* t_24[1] = {"conv11"};
	  make_convolution(net, "conv11", 1, b_24, 1, t_24, 256, 0, 1, 1, 1);

	  char* b_25[1] ={"conv11"};
	  char* t_25[1] = {"ReLU11"};
	  make_relu(net, "ReLU11", 1, b_25, 1, t_25, 0.1);

	  char* b_26[1] ={"ReLU11"};
	  char* t_26[1] = {"conv12"};
	  make_convolution(net, "conv12", 1, b_26, 1, t_26, 512, 1, 1, 3, 1);

	  char* b_27[1] ={"conv12"};
	  char* t_27[1] = {"ReLU12"};
	  make_relu(net, "ReLU12", 1, b_27, 1, t_27, 0.1);

	  char* b_28[1] ={"ReLU12"};
	  char* t_28[1] = {"conv13"};
	  make_convolution(net, "conv13", 1, b_28, 1, t_28, 256, 0, 1, 1, 1);

	  char* b_29[1] ={"conv13"};
	  char* t_29[1] = {"ReLU13"};
	  make_relu(net, "ReLU13", 1, b_29, 1, t_29, 0.1);

	  char* b_30[1] ={"ReLU13"};
	  char* t_30[1] = {"conv14"};
	  make_convolution(net, "conv14", 1, b_30, 1, t_30, 512, 1, 1, 3, 1);

	  char* b_31[1] ={"conv14"};
	  char* t_31[1] = {"ReLU14"};
	  make_relu(net, "ReLU14", 1, b_31, 1, t_31, 0.1);

	  char* b_32[1] ={"ReLU14"};
	  char* t_32[1] = {"conv15"};
	  make_convolution(net, "conv15", 1, b_32, 1, t_32, 512, 0, 1, 1, 1);

	  char* b_33[1] ={"conv15"};
	  char* t_33[1] = {"ReLU15"};
	  make_relu(net, "ReLU15", 1, b_33, 1, t_33, 0.1);

	  char* b_34[1] ={"ReLU15"};
	  char* t_34[1] = {"conv16"};
	  make_convolution(net, "conv16", 1, b_34, 1, t_34, 1024, 1, 1, 3, 1);

	  char* b_35[1] ={"conv16"};
	  char* t_35[1] = {"ReLU16"};
	  make_relu(net, "ReLU16", 1, b_35, 1, t_35, 0.1);

	  char* b_36[1] = {"ReLU16"};
	  char* t_36[1] = {"pool4"};
	  make_pooling(net, "pool4", 1, b_36, 1, t_36, 0, 2, 2, MAX, 0);

	  char* b_37[1] ={"pool4"};
	  char* t_37[1] = {"conv17"};
	  make_convolution(net, "conv17", 1, b_37, 1, t_37, 512, 0, 1, 1, 1);

	  char* b_38[1] ={"conv17"};
	  char* t_38[1] = {"ReLU17"};
	  make_relu(net, "ReLU17", 1, b_38, 1, t_38, 0.1);

	  char* b_39[1] ={"ReLU17"};
	  char* t_39[1] = {"conv18"};
	  make_convolution(net, "conv18", 1, b_39, 1, t_39, 1024, 1, 1, 3, 1);

	  char* b_40[1] ={"conv18"};
	  char* t_40[1] = {"ReLU18"};
	  make_relu(net, "ReLU18", 1, b_40, 1, t_40, 0.1);

	  char* b_41[1] ={"ReLU18"};
	  char* t_41[1] = {"conv19"};
	  make_convolution(net, "conv19", 1, b_41, 1, t_41, 512, 0, 1, 1, 1);

	  char* b_42[1] ={"conv19"};
	  char* t_42[1] = {"ReLU19"};
	  make_relu(net, "ReLU19", 1, b_42, 1, t_42, 0.1);

	  char* b_43[1] ={"ReLU19"};
	  char* t_43[1] = {"conv20"};
	  make_convolution(net, "conv20", 1, b_43, 1, t_43, 1024, 1, 1, 3, 1);

	  char* b_44[1] ={"conv20"};
	  char* t_44[1] = {"ReLU20"};
	  make_relu(net, "ReLU20", 1, b_44, 1, t_44, 0.1);

	  char* b_45[1] ={"ReLU20"};
	  char* t_45[1] = {"conv21"};
	  make_convolution(net, "conv21", 1, b_45, 1, t_45, 1024, 1, 1, 3, 1);

	  char* b_46[1] ={"conv21"};
	  char* t_46[1] = {"ReLU21"};
	  make_relu(net, "ReLU21", 1, b_46, 1, t_46, 0.1);

	  char* b_47[1] ={"ReLU21"};
	  char* t_47[1] = {"conv22"};
	  make_convolution(net, "conv22", 1, b_47, 1, t_47, 1024, 1, 2, 3, 1);

	  char* b_48[1] ={"conv22"};
	  char* t_48[1] = {"ReLU22"};
	  make_relu(net, "ReLU22", 1, b_48, 1, t_48, 0.1);

	  char* b_49[1] ={"ReLU22"};
	  char* t_49[1] = {"conv23"};
	  make_convolution(net, "conv23", 1, b_49, 1, t_49, 1024, 1, 1, 3, 1);

	  char* b_50[1] ={"conv23"};
	  char* t_50[1] = {"ReLU23"};
	  make_relu(net, "ReLU23", 1, b_50, 1, t_50, 0.1);

	  char* b_51[1] ={"ReLU23"};
	  char* t_51[1] = {"conv24"};
	  make_convolution(net, "conv24", 1, b_51, 1, t_51, 1024, 1, 1, 3, 1);

	  char* b_52[1] ={"conv24"};
	  char* t_52[1] = {"ReLU24"};
	  make_relu(net, "ReLU24", 1, b_52, 1, t_52, 0.1);

	  char* b_53[1] = {"ReLU24"};
	  char* t_53[1] = {"connect1"};
	  make_innerproduct(net, "connect1", 1, b_53, 1, t_53, 4096);

	  char* b_54[1] ={"connect1"};
	  char* t_54[1] = {"ReLU25"};
	  make_relu(net, "ReLU25", 1, b_54, 1, t_54, 0.1);

	  char* b_55[1] = {"ReLU25"};
	  char* t_55[1] = {"connect2"};
	  make_innerproduct(net, "connect2", 1, b_55, 1, t_55, 1470);

	  char* inputblobs[1] = {"data"};
	  char* outputblobs[1] = {"connect2"};
	  set_net_io(net, 1, inputblobs, 1, outputblobs);
}

void configssd(struct Net* net){
	  net->layer_num = 0;
	  net->tops_num = 0;
	  net->tops = NULL;
	  net->names = NULL;
	  net->layer_parameter = NULL;

	  char* input[1] = {"data"};
	  make_input(net, "data", 1, input, 1, 3, 500, 500);

	  char* b_1[1] = {"data"};
	  char* t_1[1] = {"conv1_1"};
	  make_convolution(net, "conv1_1", 1, b_1, 1, t_1, 64, 1, 1, 3, 1);


	  char* b_2[1] = {"conv1_1"};
	  char* t_2[1] = {"conv1_1_relu"};
	  make_relu(net, "relu1_1", 1, b_2, 1, t_2, 0);

	  char* b_3[1] = {"conv1_1_relu"};
	  char* t_3[1] = {"conv1_2"};
	  make_convolution(net, "conv1_2", 1, b_3, 1, t_3, 64, 1, 1, 3, 1);

	  char* b_4[1] = {"conv1_2"};
	  char* t_4[1] = {"conv1_2_relu"};
	  make_relu(net, "relu1_2", 1, b_4, 1, t_4, 0);

	  char* b_5[1] = {"conv1_2_relu"};
	  char* t_5[1] = {"pool1"};
	  make_pooling(net, "pool1", 1, b_5, 1, t_5, 0, 2, 2, MAX, 0);//

	  char* b_6[1] = {"pool1"};
	  char* t_6[1] = {"conv2_1"};
	  make_convolution(net, "conv2_1", 1, b_6, 1, t_6, 128, 1, 1, 3, 1);

	  char* b_7[1] = {"conv2_1"};
	  char* t_7[1] = {"conv2_1_relu"};
	  make_relu(net, "relu2_1", 1, b_7, 1, t_7, 0);

	  char* b_8[1] = {"conv2_1_relu"};
	  char* t_8[1] = {"conv2_2"};
	  make_convolution(net, "conv2_2", 1, b_8, 1, t_8, 128, 1, 1, 3, 1);

	  char* b_9[1] = {"conv2_2"};
	  char* t_9[1] = {"conv2_2_relu"};
	  make_relu(net, "relu2_2", 1, b_9, 1, t_9, 0);

	  char* b_10[1] = {"conv2_2_relu"};
	  char* t_10[1] = {"pool2"};
	  make_pooling(net, "pool2", 1, b_10, 1, t_10, 0, 2, 2, MAX, 0);

	  char* b_11[1] = {"pool2"};
	  char* t_11[1] = {"conv3_1"};
	  make_convolution(net, "conv3_1", 1, b_11, 1, t_11, 256, 1, 1, 3, 1);

	  char* b_12[1] = {"conv3_1"};
	  char* t_12[1] = {"conv3_1_relu"};
	  make_relu(net, "relu3_1", 1, b_12, 1, t_12, 0);

	  char* b_13[1] = {"conv3_1_relu"};
	  char* t_13[1] = {"conv3_2"};
	  make_convolution(net, "conv3_2", 1, b_13, 1, t_13, 256, 1, 1, 3, 1);

	  char* b_14[1] = {"conv3_2"};
	  char* t_14[1] = {"conv3_2_relu"};
	  make_relu(net, "relu3_2", 1, b_14, 1, t_14, 0);

	  char* b_15[1] = {"conv3_2_relu"};
	  char* t_15[1] = {"conv3_3"};
	  make_convolution(net, "conv3_3", 1, b_15, 1, t_15, 256, 1, 1, 3, 1);

	  char* b_16[1] = {"conv3_3"};
	  char* t_16[1] = {"conv3_3_relu"};
	  make_relu(net, "relu3_3", 1, b_16, 1, t_16, 0);

	  char* b_17[1] = {"conv3_3_relu"};
	  char* t_17[1] = {"pool3"};
	  make_pooling(net, "pooling3", 1, b_17, 1, t_17, 0, 2, 2, MAX, 0);

	  char* b_18[1] = {"pool3"};
	  char* t_18[1] = {"conv4_1"};
	  make_convolution(net, "conv4_1", 1, b_18, 1, t_18, 512, 1, 1, 3, 1);

	  char* b_19[1] = {"conv4_1"};
	  char* t_19[1] = {"conv4_1_relu"};
	  make_relu(net, "relu4_1", 1, b_19, 1, t_19, 0);

	  char* b_20[1] = {"conv4_1_relu"};
	  char* t_20[1] = {"conv4_2"};
	  make_convolution(net, "conv4_2", 1, b_20, 1, t_20, 512, 1, 1, 3, 1);

	  char* b_21[1] = {"conv4_2"};
	  char* t_21[1] = {"conv4_2_relu"};
	  make_relu(net, "relu4_2", 1, b_21, 1, t_21, 0);

	  char* b_22[1] = {"conv4_2_relu"};
	  char* t_22[1] = {"conv4_3"};
	  make_convolution(net, "conv4_3", 1, b_22, 1, t_22, 512, 1, 1, 3, 1);

	  char* b_23[1] = {"conv4_3"};
	  char* t_23[1] = {"conv4_3_relu"};
	  make_relu(net, "relu4_3", 1, b_23, 1, t_23, 0);

	  char* b_24[1] = {"conv4_3_relu"};
	  char* t_24[1] = {"pool4"};
	  make_pooling(net, "pool4", 1, b_24, 1, t_24, 0, 2, 2, MAX, 0);

	  char* b_25[1] = {"pool4"};
	  char* t_25[1] = {"conv5_1"};
	  make_convolution(net, "conv5_1", 1, b_25, 1, t_25, 512, 1, 1, 3, 1);

	  char* b_26[1] = {"conv5_1"};
	  char* t_26[1] = {"conv5_1_relu"};
	  make_relu(net, "relu5_1", 1, b_26, 1, t_26, 0);

	  char* b_27[1] = {"conv5_1_relu"};
	  char* t_27[1] = {"conv5_2"};
	  make_convolution(net, "conv5_2", 1, b_27, 1, t_27, 512, 1, 1, 3, 1);

	  char* b_28[1] = {"conv5_2"};
	  char* t_28[1] = {"conv5_2_relu"};
	  make_relu(net, "relu5_2", 1, b_28, 1, t_28, 0);

	  char* b_29[1] = {"conv5_2_relu"};
	  char* t_29[1] = {"conv5_3"};
	  make_convolution(net, "conv5_3", 1, b_29, 1, t_29, 512, 1, 1, 3, 1);

	  char* b_30[1] = {"conv5_3"};
	  char* t_30[1] = {"conv5_3_relu"};
	  make_relu(net, "relu5_3", 1, b_30, 1, t_30, 0);

	  char* b_31[1] = {"conv5_3_relu"};
	  char* t_31[1] = {"pool5"};
	  make_pooling(net, "pool5", 1, b_31, 1, t_31, 1, 1, 3, MAX, 0);

	  char* b_32[1] = {"pool5"};
	  char* t_32[1] = {"fc6"};
	  make_convolution(net, "fc6", 1, b_32, 1, t_32, 1024, 6, 1, 3, 6);

	  char* b_33[1] = {"fc6"};
	  char* t_33[1] = {"fc6_relu"};
	  make_relu(net, "relu6", 1, b_33, 1, t_33, 0);

	  char* b_34[1] = {"fc6_relu"};
	  char* t_34[1] = {"fc7"};
	  make_convolution(net, "fc7", 1, b_34, 1, t_34, 1024, 0, 1, 1, 1);

	  char* b_35[1] = {"fc7"};
	  char* t_35[1] = {"fc7_relu"};
	  make_relu(net, "relu7", 1, b_35, 1, t_35, 0);

	  char* b_36[1] = {"fc7_relu"};
	  char* t_36[1] = {"conv6_1"};
	  make_convolution(net, "conv6_1", 1, b_36, 1, t_36, 256, 0, 1, 1, 1);

	  char* b_37[1] = {"conv6_1"};
	  char* t_37[1] = {"conv6_1_relu"};
	  make_relu(net, "conv6_1_relu", 1, b_37, 1, t_37, 0);

	  char* b_38[1] = {"conv6_1_relu"};
	  char* t_38[1] = {"conv6_2"};
	  make_convolution(net, "conv6_2", 1, b_38, 1, t_38, 512, 1, 2, 3, 1);

	  char* b_39[1] = {"conv6_2"};
	  char* t_39[1] = {"conv6_2_relu"};
	  make_relu(net, "conv6_2_relu", 1, b_39, 1, t_39, 0);

	  char* b_40[1] = {"conv6_2_relu"};
	  char* t_40[1] = {"conv7_1"};
	  make_convolution(net, "conv7_1", 1, b_40, 1, t_40, 128, 0, 1, 1, 1);

	  char* b_41[1] = {"conv7_1"};
	  char* t_41[1] = {"conv7_1_relu"};
	  make_relu(net, "conv7_1_relu", 1, b_41, 1, t_41, 0);

	  char* b_42[1] = {"conv7_1_relu"};
	  char* t_42[1] = {"conv7_2"};
	  make_convolution(net, "conv7_2", 1, b_42, 1, t_42, 256, 1, 2, 3, 1);

	  char* b_43[1] = {"conv7_2"};
	  char* t_43[1] = {"conv7_2_relu"};
	  make_relu(net, "conv7_2_relu", 1, b_43, 1, t_43, 0);

	  char* b_44[1] = {"conv7_2_relu"};
	  char* t_44[1] = {"conv8_1"};
	  make_convolution(net, "conv8_1", 1, b_44, 1, t_44, 128, 0, 1, 1, 1);

	  char* b_45[1] = {"conv8_1"};
	  char* t_45[1] = {"conv8_1_relu"};
	  make_relu(net, "conv8_2_relu", 1, b_45, 1, t_45, 0);

	  char* b_46[1] = {"conv8_1_relu"};
	  char* t_46[1] = {"conv8_2"};
	  make_convolution(net, "conv8_2", 1, b_46, 1, t_46, 256, 1, 2, 3, 1);

	  char* b_47[1] = {"conv8_2"};
	  char* t_47[1] = {"conv8_2_relu"};
	  make_relu(net, "conv8_2_relu", 1, b_47, 1, t_47, 0);

	  char* b_48[1] = {"conv8_2_relu"};
	  char* t_48[1] = {"conv9_1"};
	  make_convolution(net, "conv9_1", 1, b_48, 1, t_48, 128, 0, 1, 1, 1);

	  char* b_49[1] = {"conv9_1"};
	  char* t_49[1] = {"conv9_1_relu"};
	  make_relu(net, "conv9_1", 1, b_49, 1, t_49, 0);

	  char* b_50[1] = {"conv9_1_relu"};
	  char* t_50[1] = {"conv9_2"};
	  make_convolution(net, "conv9_2", 1, b_50, 1, t_50, 256, 1, 2, 3, 1);

	  char* b_51[1] = {"conv9_2"};
	  char* t_51[1] = {"conv9_2_relu"};
	  make_relu(net, "conv9_2_relu", 1, b_51, 1, t_51, 0);

	  char* b_52[1] = {"conv9_2_relu"};
	  char* t_52[1] = {"pool6"};
	  make_pooling(net, "pool6", 1, b_52, 1, t_52, 0, 0, 0, AVE, 1);

	  char* b_53[1] = {"conv4_3_relu"};
	  char* t_53[1] = {"conv4_3_norm"};
	  make_normalize(net, "conv4_3_norm", 1, b_53, 1, t_53, 0, 0);

	  char* b_54[1] = {"conv4_3_norm"};
	  char* t_54[1] = {"conv4_3_norm_mbox_loc"};
	  make_convolution(net, "conv4_3_norm_mbox_loc", 1, b_54, 1, t_54, 12, 1, 1, 3, 1);

	  char* b_55[1] = {"conv4_3_norm_mbox_loc"};
	  char* t_55[1] = {"conv4_3_norm_mbox_loc_perm"};
	  make_permute(net, "conv4_3_norm_mbox_loc_perm", 1, b_55, 1, t_55, 0, 2, 3, 1);

	  char* b_56[1] = {"conv4_3_norm_mbox_loc_perm"};
	  char* t_56[1] = {"conv4_3_norm_mbox_loc_flat"};
	  make_flatten(net, "conv4_3_norm_mbox_loc_flat", 1, b_56, 1, t_56, 1, 3);

	  char* b_57[1] = {"conv4_3_norm"};
	  char* t_57[1] = {"conv4_3_norm_mbox_conf"};
	  make_convolution(net, "conv4_3_norm_mbox_conf", 1, b_57, 1, t_57, 63, 1, 1, 3, 1);

	  char* b_58[1] = {"conv4_3_norm_mbox_conf"};
	  char* t_58[1] = {"conv4_3_norm_mbox_conf_perm"};
	  make_permute(net, "conv4_3_norm_mbox_conf_perm", 1, b_58, 1, t_58, 0, 2, 3, 1);

	  char* b_59[1] = {"conv4_3_norm_mbox_conf_perm"};
	  char* t_59[1] = {"conv4_3_norm_mbox_conf_flat"};
	  make_flatten(net, "conv4_3_norm_mbox_conf_flat", 1, b_59, 1, t_59, 1, 3);

	  char* b_60[2] = {"conv4_3_norm","data"};
	  char* t_60[1] = {"conv4_3_norm_mbox_priorbox"};
	  make_priorbox(net, "conv4_3_norm_mbox_priorbox", 2, b_60, 1, t_60, 35, -1, 2.0, -1.0, 0.1, 0.1, 0.2, 0.2);

	  char* b_61[1] = {"fc7_relu"};
	  char* t_61[1] = {"fc7_mbox_loc"};
	  make_convolution(net, "fc7_mbox_loc", 1, b_61, 1, t_61, 24, 1, 1, 3, 1);

	  char* b_62[1] = {"fc7_mbox_loc"};
	  char* t_62[1] = {"fc7_mbox_loc_perm"};
	  make_permute(net, "fc7_mbox_loc_perm", 1, b_62, 1, t_62, 0, 2, 3, 1);

	  char* b_63[1] = {"fc7_mbox_loc_perm"};
	  char* t_63[1] = {"fc7_mbox_loc_flat"};
	  make_flatten(net, "fc7_mbox_loc_flat", 1, b_63, 1, t_63, 1, 3);

	  char* b_64[1] = {"fc7_relu"};
	  char* t_64[1] = {"fc7_mbox_conf"};
	  make_convolution(net, "fc7_mbox_conf", 1, b_64, 1, t_64, 126, 1, 1, 3, 1);

	  char* b_65[1] = {"fc7_mbox_conf"};
	  char* t_65[1] = {"fc7_mbox_conf_perm"};
	  make_permute(net, "fc7_mbox_conf_perm", 1, b_65, 1, t_65, 0, 2, 3, 1);

	  char* b_66[1] = {"fc7_mbox_conf_perm"};
	  char* t_66[1] = {"fc7_mbox_conf_flat"};
	  make_flatten(net, "fc7_mbox_conf_flat", 1, b_66, 1, t_66, 1, 3);

	  char* b_67[2] = {"fc7_relu", "data"};
	  char* t_67[1] = {"fc7_mbox_priorbox"};
	  make_priorbox(net, "fc7_mbox_priorbox", 2, b_67, 1, t_67, 75.0, 155.0, 2, 3, 0.1, 0.1, 0.2, 0.2);

	  char* b_68[1] = {"conv6_2_relu"};
	  char* t_68[1] = {"conv6_2_mbox_loc"};
	  make_convolution(net, "conv6_2_mbox_loc", 1, b_68, 1, t_68, 24, 1, 1, 3, 1);

	  char* b_69[1] = {"conv6_2_mbox_loc"};
	  char* t_69[1] = {"conv6_2_mbox_loc_perm"};
	  make_permute(net, "conv6_2_mbox_loc_perm", 1, b_69, 1, t_69, 0, 2, 3, 1);

	  char* b_70[1] = {"conv6_2_mbox_loc_perm"};
	  char* t_70[1] = {"conv6_2_mbox_loc_flat"};
	  make_flatten(net, "conv6_2_mbox_loc_flat", 1, b_70, 1, t_70, 1, 3);

	  char* b_71[1] = {"conv6_2_relu"};
	  char* t_71[1] = {"conv6_2_mbox_conf"};
	  make_convolution(net, "conv6_2_mbox_conf", 1, b_71, 1, t_71, 126, 1, 1, 3, 1);

	  char* b_72[1] = {"conv6_2_mbox_conf"};
	  char* t_72[1] = {"conv6_2_mbox_conf_perm"};
	  make_permute(net, "conv6_2_mbox_conf_perm", 1, b_72, 1, t_72, 0, 2, 3, 1);

	  char* b_73[1] = {"conv6_2_mbox_conf_perm"};
	  char* t_73[1] = {"conv6_2_mbox_conf_flat"};
	  make_flatten(net, "conv6_2_mbox_conf_flat", 1, b_73, 1, t_73, 1, 3);

	  char* b_74[2] = {"conv6_2_relu", "data"};
	  char* t_74[1] = {"conv6_2_mbox_priorbox"};
	  make_priorbox(net, "conv6_2_mbox_priorbox", 2, b_74, 1, t_74, 155.0, 235.0, 2, 3, 0.1, 0.1, 0.2, 0.2);

	  char* b_75[1] = {"conv7_2_relu"};
	  char* t_75[1] = {"conv7_2_mbox_loc"};
	  make_convolution(net, "conv7_2_mbox_loc", 1, b_75, 1, t_75, 24, 1, 1, 3, 1);

	  char* b_76[1] = {"conv7_2_mbox_loc"};
	  char* t_76[1] = {"conv7_2_mbox_loc_perm"};
	  make_permute(net, "conv7_2_mbox_loc_perm", 1, b_76, 1, t_76, 0, 2, 3, 1);

	  char* b_77[1] = {"conv7_2_mbox_loc_perm"};
	  char* t_77[1] = {"conv7_2_mbox_loc_flat"};
	  make_flatten(net, "conv7_2_mbox_loc_flat", 1, b_77, 1, t_77, 1, 3);

	  char* b_78[1] = {"conv7_2_relu"};
	  char* t_78[1] = {"conv7_2_mbox_conf"};
	  make_convolution(net, "conv7_2_mbox_conf", 1, b_78, 1, t_78, 126, 1, 1, 3, 1);

	  char* b_79[1] = {"conv7_2_mbox_conf"};
	  char* t_79[1] = {"conv7_2_mbox_conf_perm"};
	  make_permute(net, "conv7_2_mbox_conf_perm", 1, b_79, 1, t_79, 0, 2, 3, 1);

	  char* b_80[1] = {"conv7_2_mbox_conf_perm"};
	  char* t_80[1] = {"conv7_2_mbox_conf_flat"};
	  make_flatten(net, "conv7_2_mbox_conf_flat", 1, b_80, 1, t_80, 1, 3);

	  char* b_81[2] = {"conv7_2_relu", "data"};
	  char* t_81[1] = {"conv7_2_mbox_priorbox"};
	  make_priorbox(net, "conv7_2_mbox_priorbox", 2, b_81, 1, t_81, 235, 315, 2, 3, 0.1, 0.1, 0.2, 0.2);

	  char* b_82[1] = {"conv8_2_relu"};
	  char* t_82[1] = {"conv8_2_mbox_loc"};
	  make_convolution(net, "conv8_2_mbox_loc", 1, b_82, 1, t_82, 24, 1, 1, 3, 1);

	  char* b_83[1] = {"conv8_2_mbox_loc"};
	  char* t_83[1] = {"conv8_2_mbox_loc_perm"};
	  make_permute(net, "conv8_2_mbox_loc_perm", 1, b_83, 1, t_83, 0, 2, 3, 1);

	  char* b_84[1] = {"conv8_2_mbox_loc_perm"};
	  char* t_84[1] = {"conv8_2_mbox_loc_flat"};
	  make_flatten(net, "conv8_2_mbox_loc_flat", 1, b_84, 1, t_84, 1, 3);

	  char* b_85[1] = {"conv8_2_relu"};
	  char* t_85[1] = {"conv8_2_mbox_conf"};
	  make_convolution(net, "conv8_2_mbox_conf", 1, b_85, 1, t_85, 126, 1, 1, 3, 1);

	  char* b_86[1] = {"conv8_2_mbox_conf"};
	  char* t_86[1] = {"conv8_2_mbox_conf_perm"};
	  make_permute(net, "conv8_2_mbox_conf_perm", 1, b_86, 1, t_86, 0, 2, 3, 1);

	  char* b_87[1] = {"conv8_2_mbox_conf_perm"};
	  char* t_87[1] = {"conv8_2_mbox_conf_flat"};
	  make_flatten(net, "conv8_2_mbox_conf_flat", 1, b_87, 1, t_87, 1, 3);

	  char* b_88[2] = {"conv8_2_relu", "data"};
	  char* t_88[1] = {"conv8_2_mbox_priorbox"};
	  make_priorbox(net, "conv8_2_mbox_priorbox", 2, b_88, 1, t_88, 315, 395, 2, 3, 0.1, 0.1, 0.2, 0.2);

	  char* b_89[1] = {"conv9_2_relu"};
	  char* t_89[1] = {"conv9_2_mbox_loc"};
	  make_convolution(net, "conv9_2_mbox_loc", 1, b_89, 1, t_89, 24, 1, 1, 3, 1);

	  char* b_90[1] = {"conv9_2_mbox_loc"};
	  char* t_90[1] = {"conv9_2_mbox_loc_perm"};
	  make_permute(net, "conv9_2_mbox_loc_perm", 1, b_90, 1, t_90, 0, 2, 3, 1);

	  char* b_91[1] = {"conv9_2_mbox_loc_perm"};
	  char* t_91[1] = {"conv9_2_mbox_loc_flat"};
	  make_flatten(net, "conv9_2_mbox_loc_flat", 1, b_91, 1, t_91, 1, 3);

	  char* b_92[1] = {"conv9_2_relu"};
	  char* t_92[1] = {"conv9_2_mbox_conf"};
	  make_convolution(net, "conv9_2_mbox_conf", 1, b_92, 1, t_92, 126, 1, 1, 3, 1);

	  char* b_93[1] = {"conv9_2_mbox_conf"};
	  char* t_93[1] = {"conv9_2_mbox_conf_perm"};
	  make_permute(net, "conv9_2_mbox_conf_perm", 1, b_93, 1, t_93, 0, 2, 3, 1);

	  char* b_94[1] = {"conv9_2_mbox_conf_perm"};
	  char* t_94[1] = {"conv9_2_mbox_conf_flat"};
	  make_flatten(net, "conv9_2_mbox_conf_flat", 1, b_94, 1, t_94, 1, 3);

	  char* b_95[2] = {"conv9_2_relu", "data"};
	  char* t_95[1] = {"conv9_2_mbox_priorbox"};
	  make_priorbox(net, "conv9_2_mbox_priorbox", 2, b_95, 1, t_95, 395, 475, 2, 3, 0.1, 0.1, 0.2, 0.2);

	  char* b_96[1] = {"pool6"};
	  char* t_96[1] = {"pool6_mbox_loc"};
	  make_convolution(net, "pool6_mbox_loc", 1, b_96, 1, t_96, 24, 1, 1, 3, 1);

	  char* b_97[1] = {"pool6_mbox_loc"};
	  char* t_97[1] = {"pool6_mbox_loc_perm"};
	  make_permute(net, "pool6_mbox_loc_perm", 1, b_97, 1, t_97, 0, 2, 3, 1);

	  char* b_98[1] = {"pool6_mbox_loc_perm"};
	  char* t_98[1] = {"pool6_mbox_loc_flat"};
	  make_flatten(net, "pool6_mbox_loc_flat", 1, b_98, 1, t_98, 1, 3);

	  char* b_99[1]= {"pool6"};
	  char* t_99[1]= {"pool6_mbox_conf"};
	  make_convolution(net, "pool6_mbox_conf", 1, b_99, 1, t_99, 126, 1, 1, 3, 1);

	  char* b_100[1] = {"pool6_mbox_conf"};
	  char* t_100[1] = {"pool6_mbox_conf_perm"};
	  make_permute(net, "pool6_mbox_conf_perm", 1, b_100, 1, t_100, 0, 2, 3, 1);

	  char* b_101[1] = {"pool6_mbox_conf_perm"};
	  char* t_101[1] = {"pool6_mbox_conf_flat"};
	  make_flatten(net, "pool6_mbox_conf_flat", 1, b_101, 1, t_101, 1, 3);

	  char* b_102[2] = {"pool6", "data"};
	  char* t_102[1] = {"pool6_mbox_priorbox"};
	  make_priorbox(net, "pool6_mbox_priorbox", 2, b_102, 1, t_102, 475, 555, 2, 3, 0.1, 0.1, 0.2, 0.2);

	  char* b_103[7] = {"conv4_3_norm_mbox_loc_flat", "fc7_mbox_loc_flat", "conv6_2_mbox_loc_flat", "conv7_2_mbox_loc_flat", "conv8_2_mbox_loc_flat" ,"conv9_2_mbox_loc_flat", "pool6_mbox_loc_flat"};
	  char* t_103[1] = {"mbox_loc"};
	  make_concat(net, "mbox_loc", 7, b_103, 1, t_103, 1);

	  char* b_104[7] = {"conv4_3_norm_mbox_conf_flat", "fc7_mbox_conf_flat", "conv6_2_mbox_conf_flat", "conv7_2_mbox_conf_flat", "conv8_2_mbox_conf_flat" ,"conv9_2_mbox_conf_flat", "pool6_mbox_conf_flat"};
	  char* t_104[1] = {"mbox_conf"};
	  make_concat(net, "mbox_conf", 7, b_104, 1, t_104, 1);

	  char* b_105[7] = {"conv4_3_norm_mbox_priorbox", "fc7_mbox_priorbox", "conv6_2_mbox_priorbox", "conv7_2_mbox_priorbox", "conv8_2_mbox_priorbox" ,"conv9_2_mbox_priorbox", "pool6_mbox_priorbox"};
	  char* t_105[1] = {"mbox_priorbox"};
	  make_concat(net, "mbox_priorbox", 7, b_105, 1, t_105, 2);

	  char* b_106[1] = {"mbox_conf"};
	  char* t_106[1] = {"mbox_conf_reshape"};
	  make_reshape(net, "mbox_conf_reshape", 1, b_106, 1, t_106, 0, -1, 21, 0);

	  char* b_107[1] = {"mbox_conf_reshape"};
	  char* t_107[1] = {"mbox_conf_softmax"};
	  make_softmax(net, "mbox_conf_softmax", 1, b_107, 1, t_107, 2);

	  char* b_108[1] = {"mbox_conf_softmax"};
	  char* t_108[1] = {"mbox_conf_flatten"};
	  make_flatten(net, "mbox_conf_flatten", 1, b_108, 1, t_108, 1, 3);

	  char* inputblobs[1] = {"data"};
	  char* outputblobs[3] = {"mbox_conf_flatten" ,"mbox_loc", "mbox_priorbox"};
	  //char* outputblobs[2] = {"mbox_conf_flatten" ,"mbox_loc"};
	  //char* outputblobs[1] = {"conv4_3_norm_mbox_loc_perm"};
	  //char* outputblobs[2] = {"conv4_3_norm", "pool6"};
	  set_net_io(net, 1, inputblobs, 3, outputblobs);
}
int main(){
  int i;
  double start, finish;
  struct Net net;
  cublasStatus_t status;
  status = cublasCreate(&(net.cublas_handle));
  if(status != CUBLAS_STATUS_SUCCESS){
	  printf("get cublas handle error!\n");
  }else{
	  printf("get cublas handle success!\n");
  }
  //struct LayerName names;
  //InitLayerName(&names);
  struct Map_Type_Setup_Forward handle;
  net.handle.num = 0;
  net.handle.functions = NULL;
  regist_layer(&(net.handle));
  configssd(&net);
  //confignet(&net);
  setup(net);
    printf("network input and ouput\n");
    for(i = 0; i < net.input_num; ++i)
  	  printf("input%d (%d,%d,%d,%d)\n", i+1, net.inputs[i]->n, net.inputs[i]->c, net.inputs[i]->h, net.inputs[i]->w);
    for(i = 0; i < net.output_num; ++i)
  	  printf("output%d (%d,%d,%d,%d)\n", i+1, net.outputs[i]->n, net.outputs[i]->c, net.outputs[i]->h, net.outputs[i]->w);
//  ssd
   char weightpath[200];
   FILE* fp = NULL;
   float* data;
   int j;
   for(i = 0; i < net.layer_num; ++i){
 	  struct LayerParameter param = net.layer_parameter[i];
 	  if(param.has_learn_parameter){
 		  sprintf(weightpath, "%s/%s.txt","./fold_for_test/weights", param.name);
 		  fp = fopen(weightpath, "r");
 		  if(fp == NULL){
 			  printf("%s load error!\n", weightpath);
 			  for(;;);
 		  }else{
 			  for(j = 0; j < param.has_learn_parameter; ++j){
 				  data = (float*)malloc(sizeof(float)*net.layer_parameter[i].learn_parameter[j]->count);
 				  fread(data, sizeof(float), net.layer_parameter[i].learn_parameter[j]->count ,fp);
 				  cudaMemcpy(net.layer_parameter[i].learn_parameter[j]->gpu_data, data, sizeof(float)*net.layer_parameter[i].learn_parameter[j]->count, cudaMemcpyHostToDevice);
 				  free(data);
 			  }
 			  fclose(fp);
 			  printf("%s has loaded!\n", weightpath);
 		  }
//
 	  }
   }
//ssd
   data = (float*)malloc(sizeof(float)*500*3*500);
   for(i = 0; i < 500*500*3; ++i){
 	  data[i] = i%31;
   }
   cudaMemcpy(net.inputs[0]->gpu_data, data, sizeof(float)*500*500*3, cudaMemcpyHostToDevice);

   printf("start count time\n");
	start = clock();
	forward(net);//这个函数代替下面的注释语句
	finish = clock();
	printf( "%f seconds\n",(finish - start) / CLOCKS_PER_SEC);
   fp = NULL;
     char path[200];
     int count = 0;
     for(i = 0; i < net.output_num; ++i){
   	  printf("output %d\n", i);
   	  sprintf(path, "%s/output%d.txt","./fold_for_test/outputs",i);
   	  fp = fopen(path, "r");
   	  count = net.outputs[i]->count;
   	  data = (float*)malloc(sizeof(float)*count);
   	  fread(data, sizeof(float), count, fp);
   	  fclose(fp);
   	  float* data_gpu = (float*)malloc(sizeof(float)*count);
   	  cudaMemcpy(data_gpu, net.outputs[i]->gpu_data, (sizeof(float)*count), cudaMemcpyDeviceToHost);

   	  for(j = 0; j < count; ++j){
   		  if(data[j] - data_gpu[j] > 0.001)
   			  printf("out %d %d %f %f %f\n",i, j, data[j], data_gpu[j], data[j] - data_gpu[j]);
   	  }
   	  free(data_gpu);
   	  free(data);
     }
     printf("network input and ouput\n");
     for(i = 0; i < net.input_num; ++i)
   	  printf("input%d (%d,%d,%d,%d)\n", i+1, net.inputs[i]->n, net.inputs[i]->c, net.inputs[i]->h, net.inputs[i]->w);
     for(i = 0; i < net.output_num; ++i)
   	  printf("output%d (%d,%d,%d,%d)\n", i+1, net.outputs[i]->n, net.outputs[i]->c, net.outputs[i]->h, net.outputs[i]->w);




//  printf("network input and ouput\n");
//  for(i = 0; i < net.input_num; ++i)
//	  printf("input%d (%d,%d,%d,%d)\n", i+1, net.inputs[i]->n, net.inputs[i]->c, net.inputs[i]->h, net.inputs[i]->w);
//  for(i = 0; i < net.output_num; ++i)
//	  printf("output%d (%d,%d,%d,%d)\n", i+1, net.outputs[i]->n, net.outputs[i]->c, net.outputs[i]->h, net.outputs[i]->w);
//  printf("\n");
//  float temp[750000];
//  for(i = 0; i < net.inputs[0]->count; ++i){
//	  temp[i] = i%4;//i%20-10;
//  }
//  cudaMemcpy(net.inputs[0]->gpu_data, temp, sizeof(float)*net.inputs[0]->count, cudaMemcpyHostToDevice);
//  float w_temp[3] = {1,1,1};
//  float b_temp=0.5;
//  cudaMemcpy(net.layer_parameter[4].learn_parameter[0]->gpu_data, w_temp, sizeof(float)*3, cudaMemcpyHostToDevice);
//  cudaMemcpy(net.layer_parameter[4].learn_parameter[1]->gpu_data, &b_temp, sizeof(float), cudaMemcpyHostToDevice);
//    forward(net);
//  float temp1[750000];
//  cudaMemcpy(temp1, net.outputs[0]->gpu_data, sizeof(float)*net.outputs[0]->count, cudaMemcpyDeviceToHost);
//  for(i = 0; i < 100; ++i){
//	  printf("%d  %f  %f\n", i, temp[i], temp1[i]);
//  }


/*
   FILE* fp;
   int conv = 0;
   int connect = 0;
   char weights_path[100];
   float* data;
   for(i = 0; i < net.layer_num; ++i){
 	  struct LayerParameter temp = net.layer_parameter[i];
 	  if(temp.learn_parameter){
 		  if(temp.type == CONVOLUTION)
 			  sprintf(weights_path, "/home/yang/yolo-weights/convolution%d.txt", ++conv);
 		  else
 			  sprintf(weights_path, "/home/yang/yolo-weights/connect%d.txt", ++connect);
 		  printf("%s\n",weights_path);
 		  fp = fopen(weights_path,"r");
 		  data = (float*)malloc(sizeof(float)*net.layer_parameter[i].learn_parameter[1]->count);
 		  fread(data, sizeof(float), net.layer_parameter[i].learn_parameter[1]->count, fp);
 		  cudaMemcpy(net.layer_parameter[i].learn_parameter[1]->gpu_data, data, sizeof(float)*net.layer_parameter[i].learn_parameter[1]->count, cudaMemcpyHostToDevice);
 		  free(data);
 		  data = (float*)malloc(sizeof(float)*net.layer_parameter[i].learn_parameter[0]->count);
 		  fread(data, sizeof(float), net.layer_parameter[i].learn_parameter[0]->count, fp);
 		  cudaMemcpy(net.layer_parameter[i].learn_parameter[0]->gpu_data, data, sizeof(float)*net.layer_parameter[i].learn_parameter[0]->count, cudaMemcpyHostToDevice);
 		  free(data);
 		  fclose(fp);
 	  }
   }
   //加载输入数据
   fp = fopen("/home/yang/Desktop/image_data.txt","r");
   data = (float*)malloc(sizeof(float)*net.inputs[0]->count);
   fread(data, sizeof(float), net.inputs[0]->count, fp);
   cudaMemcpy(net.inputs[0]->gpu_data, data, sizeof(float)*net.inputs[0]->count, cudaMemcpyHostToDevice);
   //cudaMemcpy(net.inputs[0]->gpu_data+net.inputs[0]->count/2, data, sizeof(float)*net.inputs[0]->count/2, cudaMemcpyHostToDevice);
   free(data);
   fclose(fp);
//   start = clock();
//   cudaMalloc(&data, sizeof(float)*500*500*3*21);
//   cudaFree(data);
//   finish = clock();
//   printf( "%f seconds\n",(finish - start) / CLOCKS_PER_SEC);
   printf("start count time\n");
   start = clock();
   forward(net);//这个函数代替下面的注释语句
   finish = clock();
   printf( "%f seconds\n",(finish - start) / CLOCKS_PER_SEC);
   //输出结果对比
   float con_data[1470];
   fp = fopen("/home/yang/Desktop/conv_data.txt","r");
   fread(con_data, sizeof(float), 1470, fp);
   fclose(fp);
   data = (float*)malloc(sizeof(float)*net.outputs[0]->count);
   cudaMemcpy(data, net.outputs[0]->gpu_data, sizeof(float)*net.outputs[0]->count, cudaMemcpyDeviceToHost);
//   for(i = 0; i < 1470; ++i)
//	   //printf("%d %f %f %f %f\n", i , data[i+1470], data[i], con_data[i], data[i]-con_data[i]);
//	   printf("%d %f %f %f\n",i, data[i], con_data[i], data[i]-con_data[i]);
   printf( "%f seconds\n",(finish - start) / CLOCKS_PER_SEC);*/
   deletenet(net);
  //释放内存
  //DeleteLayerName(&names);
  //释放内存结束
   return 1;
}
#endif

#ifdef YOLO
#include <stdio.h>
#include "DL/util/net.h"
#include "DL/util/layer_factory.h"
#include <cuda_runtime.h>
#include <time.h>

void confignet(struct Net* net){
	  net->layer_num = 0;
	  net->tops_num = 0;
	  net->tops = NULL;
	  net->names = NULL;
	  net->layer_parameter = NULL;

	  char* input[1] = {"data"};
	  make_input(net, "data", 1, input, 1, 3, 448, 448);

	  char* b_1[1] = {"data"};
	  char* t_1[1] = {"conv1"};
	  make_convolution(net, "conv1", 1, b_1, 1, t_1, 64, 3, 2, 7, 1);

	  char* b_2[1] = {"conv1"};
	  char* t_2[1] = {"ReLU1"};
	  make_relu(net, "ReLU1", 1, b_2, 1, t_2, 0.1);

	  char* b_3[1] = {"ReLU1"};
	  char* t_3[1] = {"pool1"};
	  make_pooling(net, "pool1", 1, b_3, 1, t_3, 0, 2, 2, MAX, 0);

	  char* b_4[1] = {"pool1"};
	  char* t_4[1] = {"conv2"};
	  make_convolution(net, "conv2", 1, b_4, 1, t_4, 192, 1, 1, 3, 1);

	  char* b_5[1] = {"conv2"};
	  char* t_5[1] = {"ReLU2"};
	  make_relu(net, "ReLU2", 1, b_5, 1, t_5, 0.1);

	  char* b_6[1] = {"ReLU2"};
	  char* t_6[1] = {"pool2"};
	  make_pooling(net, "pool2", 1, b_6, 1, t_6, 0, 2, 2, MAX, 0);

	  char* b_7[1] = {"pool2"};
	  char* t_7[1] = {"conv3"};
	  make_convolution(net, "conv3", 1, b_7, 1, t_7, 128, 0, 1, 1, 1);

	  char* b_8[1] = {"conv3"};
	  char* t_8[1] = {"ReLU3"};
	  make_relu(net, "ReLU3", 1, b_8, 1, t_8, 0.1);

	  char* b_9[1] = {"ReLU3"};
	  char* t_9[1] = {"conv4"};
	  make_convolution(net, "conv4", 1, b_9, 1, t_9, 256, 1, 1, 3, 1);

	  char* b_10[1] = {"conv4"};
	  char* t_10[1] = {"ReLU4"};
	  make_relu(net, "ReLU3", 1, b_10, 1, t_10, 0.1);

	  char* b_11[1] = {"ReLU4"};
	  char* t_11[1] = {"conv5"};
	  make_convolution(net, "conv4", 1, b_11, 1, t_11, 256, 0, 1, 1, 1);

	  char* b_12[1] = {"conv5"};
	  char* t_12[1] = {"ReLU5"};
	  make_relu(net, "ReLU5", 1, b_12, 1, t_12, 0.1);

	  char* b_13[1] = {"ReLU5"};
	  char* t_13[1] = {"conv6"};
	  make_convolution(net, "conv4", 1, b_13, 1, t_13, 512, 1, 1, 3, 1);

	  char* b_14[1] = {"conv6"};
	  char* t_14[1] = {"ReLU6"};
	  make_relu(net, "ReLU6", 1, b_14, 1, t_14, 0.1);

	  char* b_15[1] = {"ReLU6"};
	  char* t_15[1] = {"pool3"};
	  make_pooling(net, "pool3", 1, b_15, 1, t_15, 0, 2, 2, MAX, 0);

	  char* b_16[1] = {"pool3"};
	  char* t_16[1] = {"conv7"};
	  make_convolution(net, "conv7", 1, b_16, 1, t_16, 256, 0, 1, 1, 1);

	  char* b_17[1] = {"conv7"};
	  char* t_17[1] = {"ReLU7"};
	  make_relu(net, "ReLU7", 1, b_17, 1, t_17, 0.1);

	  char* b_18[1] = {"ReLU7"};
	  char* t_18[1] = {"conv8"};
	  make_convolution(net, "conv8", 1, b_18, 1, t_18, 512, 1, 1, 3, 1);

	  char* b_19[1] = {"conv8"};
	  char* t_19[1] = {"ReLU8"};
	  make_relu(net, "ReLU8", 1, b_19, 1, t_19, 0.1);

	  char* b_20[1] = {"ReLU8"};
	  char* t_20[1] = {"conv9"};
	  make_convolution(net, "conv9", 1, b_20, 1, t_20, 256, 0, 1, 1, 1);

	  char* b_21[1] = {"conv9"};
	  char* t_21[1] = {"ReLU9"};
	  make_relu(net, "ReLU9", 1, b_21, 1, t_21, 0.1);

	  char* b_22[1] ={"ReLU9"};
	  char* t_22[1] = {"conv10"};
	  make_convolution(net, "conv10", 1, b_22, 1, t_22, 512, 1, 1, 3, 1);

	  char* b_23[1] ={"conv10"};
	  char* t_23[1] = {"ReLU10"};
	  make_relu(net, "ReLU10", 1, b_23, 1, t_23, 0.1);

	  char* b_24[1] ={"ReLU10"};
	  char* t_24[1] = {"conv11"};
	  make_convolution(net, "conv11", 1, b_24, 1, t_24, 256, 0, 1, 1, 1);

	  char* b_25[1] ={"conv11"};
	  char* t_25[1] = {"ReLU11"};
	  make_relu(net, "ReLU11", 1, b_25, 1, t_25, 0.1);

	  char* b_26[1] ={"ReLU11"};
	  char* t_26[1] = {"conv12"};
	  make_convolution(net, "conv12", 1, b_26, 1, t_26, 512, 1, 1, 3, 1);

	  char* b_27[1] ={"conv12"};
	  char* t_27[1] = {"ReLU12"};
	  make_relu(net, "ReLU12", 1, b_27, 1, t_27, 0.1);

	  char* b_28[1] ={"ReLU12"};
	  char* t_28[1] = {"conv13"};
	  make_convolution(net, "conv13", 1, b_28, 1, t_28, 256, 0, 1, 1, 1);

	  char* b_29[1] ={"conv13"};
	  char* t_29[1] = {"ReLU13"};
	  make_relu(net, "ReLU13", 1, b_29, 1, t_29, 0.1);

	  char* b_30[1] ={"ReLU13"};
	  char* t_30[1] = {"conv14"};
	  make_convolution(net, "conv14", 1, b_30, 1, t_30, 512, 1, 1, 3, 1);

	  char* b_31[1] ={"conv14"};
	  char* t_31[1] = {"ReLU14"};
	  make_relu(net, "ReLU14", 1, b_31, 1, t_31, 0.1);

	  char* b_32[1] ={"ReLU14"};
	  char* t_32[1] = {"conv15"};
	  make_convolution(net, "conv15", 1, b_32, 1, t_32, 512, 0, 1, 1, 1);

	  char* b_33[1] ={"conv15"};
	  char* t_33[1] = {"ReLU15"};
	  make_relu(net, "ReLU15", 1, b_33, 1, t_33, 0.1);

	  char* b_34[1] ={"ReLU15"};
	  char* t_34[1] = {"conv16"};
	  make_convolution(net, "conv16", 1, b_34, 1, t_34, 1024, 1, 1, 3, 1);

	  char* b_35[1] ={"conv16"};
	  char* t_35[1] = {"ReLU16"};
	  make_relu(net, "ReLU16", 1, b_35, 1, t_35, 0.1);

	  char* b_36[1] = {"ReLU16"};
	  char* t_36[1] = {"pool4"};
	  make_pooling(net, "pool4", 1, b_36, 1, t_36, 0, 2, 2, MAX, 0);

	  char* b_37[1] ={"pool4"};
	  char* t_37[1] = {"conv17"};
	  make_convolution(net, "conv17", 1, b_37, 1, t_37, 512, 0, 1, 1, 1);

	  char* b_38[1] ={"conv17"};
	  char* t_38[1] = {"ReLU17"};
	  make_relu(net, "ReLU17", 1, b_38, 1, t_38, 0.1);

	  char* b_39[1] ={"ReLU17"};
	  char* t_39[1] = {"conv18"};
	  make_convolution(net, "conv18", 1, b_39, 1, t_39, 1024, 1, 1, 3, 1);

	  char* b_40[1] ={"conv18"};
	  char* t_40[1] = {"ReLU18"};
	  make_relu(net, "ReLU18", 1, b_40, 1, t_40, 0.1);

	  char* b_41[1] ={"ReLU18"};
	  char* t_41[1] = {"conv19"};
	  make_convolution(net, "conv19", 1, b_41, 1, t_41, 512, 0, 1, 1, 1);

	  char* b_42[1] ={"conv19"};
	  char* t_42[1] = {"ReLU19"};
	  make_relu(net, "ReLU19", 1, b_42, 1, t_42, 0.1);

	  char* b_43[1] ={"ReLU19"};
	  char* t_43[1] = {"conv20"};
	  make_convolution(net, "conv20", 1, b_43, 1, t_43, 1024, 1, 1, 3, 1);

	  char* b_44[1] ={"conv20"};
	  char* t_44[1] = {"ReLU20"};
	  make_relu(net, "ReLU20", 1, b_44, 1, t_44, 0.1);

	  char* b_45[1] ={"ReLU20"};
	  char* t_45[1] = {"conv21"};
	  make_convolution(net, "conv21", 1, b_45, 1, t_45, 1024, 1, 1, 3, 1);

	  char* b_46[1] ={"conv21"};
	  char* t_46[1] = {"ReLU21"};
	  make_relu(net, "ReLU21", 1, b_46, 1, t_46, 0.1);

	  char* b_47[1] ={"ReLU21"};
	  char* t_47[1] = {"conv22"};
	  make_convolution(net, "conv22", 1, b_47, 1, t_47, 1024, 1, 2, 3, 1);

	  char* b_48[1] ={"conv22"};
	  char* t_48[1] = {"ReLU22"};
	  make_relu(net, "ReLU22", 1, b_48, 1, t_48, 0.1);

	  char* b_49[1] ={"ReLU22"};
	  char* t_49[1] = {"conv23"};
	  make_convolution(net, "conv23", 1, b_49, 1, t_49, 1024, 1, 1, 3, 1);

	  char* b_50[1] ={"conv23"};
	  char* t_50[1] = {"ReLU23"};
	  make_relu(net, "ReLU23", 1, b_50, 1, t_50, 0.1);

	  char* b_51[1] ={"ReLU23"};
	  char* t_51[1] = {"conv24"};
	  make_convolution(net, "conv24", 1, b_51, 1, t_51, 1024, 1, 1, 3, 1);

	  char* b_52[1] ={"conv24"};
	  char* t_52[1] = {"ReLU24"};
	  make_relu(net, "ReLU24", 1, b_52, 1, t_52, 0.1);

	  char* b_53[1] = {"ReLU24"};
	  char* t_53[1] = {"connect1"};
	  make_innerproduct(net, "connect1", 1, b_53, 1, t_53, 4096);

	  char* b_54[1] ={"connect1"};
	  char* t_54[1] = {"ReLU25"};
	  make_relu(net, "ReLU25", 1, b_54, 1, t_54, 0.1);

	  char* b_55[1] = {"ReLU25"};
	  char* t_55[1] = {"connect2"};
	  make_innerproduct(net, "connect2", 1, b_55, 1, t_55, 1470);

	  char* inputblobs[1] = {"data"};
	  char* outputblobs[1] = {"ReLU1"};
	  set_net_io(net, 1, inputblobs, 1, outputblobs);
}


int main(){
  int i;
  double start, finish;
  struct Net net;
  cublasStatus_t status;
  status = cublasCreate(&(net.cublas_handle));
  if(status != CUBLAS_STATUS_SUCCESS){
	  printf("get cublas handle error!\n");
  }else{
	  printf("get cublas handle success!\n");
  }
  //struct LayerName names;
  //InitLayerName(&names);
  struct Map_Type_Setup_Forward handle;
  net.handle.num = 0;
  net.handle.functions = NULL;
  regist_layer(&(net.handle));
  confignet(&net);
  setup(net);



   FILE* fp;
   int conv = 0;
   int connect = 0;
   char weights_path[100];
   float* data;
   for(i = 0; i < net.layer_num; ++i){
 	  struct LayerParameter temp = net.layer_parameter[i];
 	  if(temp.learn_parameter){
 		  if(temp.type == CONVOLUTION)
 			  sprintf(weights_path, "/home/yang/yolo-weights/convolution%d.txt", ++conv);
 		  else
 			  sprintf(weights_path, "/home/yang/yolo-weights/connect%d.txt", ++connect);
 		  printf("%s\n",weights_path);
 		  fp = fopen(weights_path,"r");
 		  data = (float*)malloc(sizeof(float)*net.layer_parameter[i].learn_parameter[1]->count);
 		  fread(data, sizeof(float), net.layer_parameter[i].learn_parameter[1]->count, fp);
 		  cudaMemcpy(net.layer_parameter[i].learn_parameter[1]->gpu_data, data, sizeof(float)*net.layer_parameter[i].learn_parameter[1]->count, cudaMemcpyHostToDevice);
 		  free(data);
 		  data = (float*)malloc(sizeof(float)*net.layer_parameter[i].learn_parameter[0]->count);
 		  fread(data, sizeof(float), net.layer_parameter[i].learn_parameter[0]->count, fp);
 		  cudaMemcpy(net.layer_parameter[i].learn_parameter[0]->gpu_data, data, sizeof(float)*net.layer_parameter[i].learn_parameter[0]->count, cudaMemcpyHostToDevice);
 		  free(data);
 		  fclose(fp);
 	  }
   }
   //加载输入数据
   fp = fopen("/home/yang/Desktop/image_data.txt","r");
   data = (float*)malloc(sizeof(float)*net.inputs[0]->count);
   fread(data, sizeof(float), net.inputs[0]->count, fp);
   cudaMemcpy(net.inputs[0]->gpu_data, data, sizeof(float)*net.inputs[0]->count, cudaMemcpyHostToDevice);
   //cudaMemcpy(net.inputs[0]->gpu_data+net.inputs[0]->count/2, data, sizeof(float)*net.inputs[0]->count/2, cudaMemcpyHostToDevice);
   free(data);
   fclose(fp);
//   start = clock();
//   cudaMalloc(&data, sizeof(float)*500*500*3*21);
//   cudaFree(data);
//   finish = clock();
//   printf( "%f seconds\n",(finish - start) / CLOCKS_PER_SEC);
   printf("start count time\n");
   start = clock();
   forward(net);//这个函数代替下面的注释语句
   finish = clock();
   printf( "%f seconds\n",(finish - start) / CLOCKS_PER_SEC);
   //输出结果对比
   float con_data[1470];
   fp = fopen("/home/yang/Desktop/conv_data.txt","r");
   fread(con_data, sizeof(float), 1470, fp);
   fclose(fp);
   data = (float*)malloc(sizeof(float)*net.outputs[0]->count);
   cudaMemcpy(data, net.outputs[0]->gpu_data, sizeof(float)*net.outputs[0]->count, cudaMemcpyDeviceToHost);
   for(i = 0; i < 1470; ++i)
	   //printf("%d %f %f %f %f\n", i , data[i+1470], data[i], con_data[i], data[i]-con_data[i]);
	   printf("%d %f %f %f\n",i, data[i], con_data[i], data[i]-con_data[i]);
   printf( "%f seconds\n",(finish - start) / CLOCKS_PER_SEC);
   deletenet(net);
  //释放内存
  //DeleteLayerName(&names);
  //释放内存结束
   return 1;
}

#endif


