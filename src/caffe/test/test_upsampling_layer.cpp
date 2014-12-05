#include <vector>
#include <algorithm>

#include "gtest/gtest.h"
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/vision_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

#include "opencv2/opencv.hpp"

namespace caffe {

template <typename TypeParam>
class UpsamplingLayerTest : public MultiDeviceTest<TypeParam>
{
	typedef typename TypeParam::Dtype Dtype;
	
protected:
	UpsamplingLayerTest():
		blob_bottom(new Blob<Dtype>(2, 3, 6, 8)),
		blob_top(new Blob<Dtype>()) {}
		
	virtual void SetUp()
	{
		Caffe::set_random_seed(1701);
		//fill the values
		FillerParameter filler_param;
		filler_param.set_value(1.);
		GaussianFiller<Dtype> filler(filler_param);
		filler.Fill(this->blob_bottom);
		blob_top_vec.push_back(blob_top);
		blob_bottom_vec.push_back(blob_bottom);
	}
	
	virtual ~UpsamplingLayerTest()
	{
		delete blob_bottom;
		delete blob_top;
	}
	
	Blob<Dtype>* const blob_bottom;
	Blob<Dtype>* const blob_top;
	vector<Blob<Dtype>*> blob_bottom_vec;
	vector<Blob<Dtype>*> blob_top_vec;
};

TYPED_TEST_CASE(UpsamplingLayerTest, TestDtypesAndDevices);

TYPED_TEST(UpsamplingLayerTest, TestSetUp)
{
	typedef typename TypeParam::Dtype Dtype;
	LayerParameter layer_param;
	
	UpsamplingParameter *upsampling_param = layer_param.mutable_upsampling_param();
	upsampling_param->set_new_height(3);
	upsampling_param->set_new_width(4);
	
	UpsamplingLayer<Dtype> layer(layer_param);
	layer.SetUp(this->blob_bottom_vec, &(this->blob_top_vec));
	EXPECT_EQ(this->blob_top->height(), 3);
	EXPECT_EQ(this->blob_top->width(), 4);
	EXPECT_EQ(this->blob_top->channels(), this->blob_bottom->channels());
	EXPECT_EQ(this->blob_top->num(), this->blob_bottom->num());
}

TYPED_TEST(UpsamplingLayerTest, TestForward)
{
	typedef typename TypeParam::Dtype Dtype;
	LayerParameter layer_param;
	
	UpsamplingParameter *upsampling_param = layer_param.mutable_upsampling_param();
	upsampling_param->set_new_height(6);
	upsampling_param->set_new_width(8);
	
	UpsamplingLayer<Dtype> layer(layer_param);
	layer.SetUp(this->blob_bottom_vec, &(this->blob_top_vec));
	
	layer.Forward(this->blob_bottom_vec, &(this->blob_top_vec));
	
	for(int i = 0; i < this->blob_bottom->num(); i++)
	{
		// each channel
		for(int j = 0; j < this->blob_bottom->channels(); j++)
		{
			// copy into cv::Mat
			cv::Mat mat(this->blob_bottom->height(), this->blob_bottom->width(), CV_64F);
			for(int r = 0; r < this->blob_bottom->height(); r++)
			{
				for(int c = 0; c < this->blob_bottom->width(); c++)
				{
					mat.at<double>(r, c) = *(this->blob_bottom->cpu_data() + this->blob_bottom->offset(i, j, r, c));
				}
			}
			
			resize(mat, mat, cv::Size(8, 6), 0, 0, cv::INTER_AREA); // input, output, width, height
			
			for(int r = 0; r < this->blob_top->height(); r++)
			{
				for(int c = 0; c < this->blob_top->width(); c++)
				{
					Dtype v_forward = *(this->blob_top->cpu_data() + this->blob_top->offset(i, j, r, c));
					Dtype v_gt = mat.at<double>(r, c);
					EXPECT_EQ(v_gt, v_forward);
				}
			}
		}
	}
}

TYPED_TEST(UpsamplingLayerTest, TestGradient)
{
	typedef typename TypeParam::Dtype Dtype;
	
	LayerParameter layer_param;
	UpsamplingParameter* upsampling_param = layer_param.mutable_upsampling_param();
	upsampling_param->set_new_height(6);
	upsampling_param->set_new_width(8);
	
	UpsamplingLayer<Dtype> layer(layer_param);
	
	GradientChecker<Dtype> checker(1e-2, 1e-2);
	checker.CheckGradientExhaustive(&layer, &(this->blob_bottom_vec), &(this->blob_top_vec));
}

TYPED_TEST(UpsamplingLayerTest, TestGradientUnfixedScaleRatio)
{
	typedef typename TypeParam::Dtype Dtype;
	
	LayerParameter layer_param;
	UpsamplingParameter* upsampling_param = layer_param.mutable_upsampling_param();
	upsampling_param->set_new_height(5);
	upsampling_param->set_new_width(5);
	
	UpsamplingLayer<Dtype> layer(layer_param);
	
	GradientChecker<Dtype> checker(1e-2, 1e-2);
	checker.CheckGradientExhaustive(&layer, &(this->blob_bottom_vec), &(this->blob_top_vec));
}

} // namespace caffe
