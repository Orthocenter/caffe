#include <vector>
#include <algorithm>

#include "gtest/gtest.h"
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/vision_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class SelectingLayerTest : public MultiDeviceTest<TypeParam> {
	typedef typename TypeParam::Dtype Dtype;
	
protected:
	SelectingLayerTest():
		blob_bottom(new Blob<Dtype>(2, 4, 3, 5)),
		blob_top(new Blob<Dtype>()) {}
		
	virtual void SetUp()
	{
		Caffe::set_random_seed(1701);
		// fill the values
		FillerParameter filler_param;
		filler_param.set_value(1.);
		GaussianFiller<Dtype> filler(filler_param);
		filler.Fill(this->blob_bottom);
		blob_top_vec.push_back(blob_top);
		blob_bottom_vec.push_back(blob_bottom);
	}
	
	virtual ~SelectingLayerTest()
	{
		delete blob_bottom;
		delete blob_top;
	}
	
	Blob<Dtype>* const blob_bottom;
	Blob<Dtype>* const blob_top;
	vector<Blob<Dtype>*> blob_bottom_vec;
	vector<Blob<Dtype>*> blob_top_vec;
	
public:
	static int Random_Caffe(int i) { return caffe_rng_rand() % i; }
};

TYPED_TEST_CASE(SelectingLayerTest, TestDtypesAndDevices);

TYPED_TEST(SelectingLayerTest, TestSetup)
{
	typedef typename TypeParam::Dtype Dtype;
	LayerParameter layer_param;
	
	SelectingParameter *selecting_param = layer_param.mutable_selecting_param();
	selecting_param->set_num_output(2);
	selecting_param->set_group_size(2);
	
	SelectingLayer<Dtype> layer(layer_param);
	layer.SetUp(this->blob_bottom_vec, &(this->blob_top_vec));
	EXPECT_EQ(this->blob_bottom->width(), this->blob_top->width());
	EXPECT_EQ(this->blob_bottom->height(), this->blob_top->height());
	EXPECT_EQ(this->blob_top->channels(), 2);
	EXPECT_EQ(this->blob_bottom->num(), this->blob_top->num());
}

TYPED_TEST(SelectingLayerTest, TestSetupOneGroup)
{
	typedef typename TypeParam::Dtype Dtype;
	LayerParameter layer_param;
	
	SelectingParameter *selecting_param = layer_param.mutable_selecting_param();
	selecting_param->set_num_output(1);
	selecting_param->set_group_size(4);
	
	SelectingLayer<Dtype> layer(layer_param);
	layer.SetUp(this->blob_bottom_vec, &(this->blob_top_vec));
	EXPECT_EQ(this->blob_bottom->width(), this->blob_top->width());
	EXPECT_EQ(this->blob_bottom->height(), this->blob_top->height());
	EXPECT_EQ(this->blob_top->channels(), 1);
	EXPECT_EQ(this->blob_bottom->num(), this->blob_top->num());
}

TYPED_TEST(SelectingLayerTest, TestForward)
{
	typedef typename TypeParam::Dtype Dtype;
	
	LayerParameter layer_param;
	SelectingParameter *selecting_param = layer_param.mutable_selecting_param();
	selecting_param->set_num_output(2);
	selecting_param->set_group_size(2);
	
	Caffe::set_random_seed(1701);
	SelectingLayer<Dtype> layer1(layer_param);
	layer1.SetUp(this->blob_bottom_vec, &(this->blob_top_vec));

	int spatial_dims = this->blob_bottom->height() * this->blob_bottom->width();
	int top_dims = this->blob_top->channels() * spatial_dims;
	int bottom_dims = this->blob_bottom->channels() * spatial_dims;

	// Generate forward blob
	layer1.Forward(this->blob_bottom_vec, &(this->blob_top_vec));
	
	vector<Dtype> sequence;
	for(int i = 0, offset = 0; i < 2; i++)
	{
		for(int j = 0; j < top_dims; j++) sequence.push_back(this->blob_top->cpu_data()[offset++]);
	}
	
	// Calculate ground truth
	Caffe::set_random_seed(1701);
	vector<int> shuffle(4);
	for(int i = 0; i < 4; i++) shuffle[i] = i;
	std::random_shuffle(shuffle.begin(), shuffle.end(), SelectingLayerTest<TypeParam>::Random_Caffe);
	
	vector<Dtype> sequence_gt;
	for(int i = 0; i < 2; i++)
	{
		for(int j = 0; j < spatial_dims; j++)
		{
			Dtype value = this->blob_bottom->cpu_data()[i * bottom_dims + shuffle[0] * spatial_dims + j] +
				this->blob_bottom->cpu_data()[i * bottom_dims + shuffle[1] * spatial_dims + j];
			sequence_gt.push_back(value);
		}
		
		for(int j = 0; j < spatial_dims; j++)
		{
			Dtype value = this->blob_bottom->cpu_data()[i * bottom_dims + shuffle[2] * spatial_dims + j] +
				this->blob_bottom->cpu_data()[i * bottom_dims + shuffle[3] * spatial_dims + j];
			sequence_gt.push_back(value);
		}
	}
	
	EXPECT_EQ(sequence.size(), sequence_gt.size());
	for(int i = 0; i < sequence.size(); i++)
	{
		EXPECT_EQ(sequence[i], sequence_gt[i]);
	}
}

TYPED_TEST(SelectingLayerTest, TestForwardOneGroup)
{
	typedef typename TypeParam::Dtype Dtype;
	
	LayerParameter layer_param;
	SelectingParameter *selecting_param = layer_param.mutable_selecting_param();
	selecting_param->set_num_output(1);
	selecting_param->set_group_size(3);
	
	Caffe::set_random_seed(1701);
	SelectingLayer<Dtype> layer1(layer_param);
	layer1.SetUp(this->blob_bottom_vec, &(this->blob_top_vec));

	int spatial_dims = this->blob_bottom->height() * this->blob_bottom->width();
	int top_dims = this->blob_top->channels() * spatial_dims;
	int bottom_dims = this->blob_bottom->channels() * spatial_dims;

	// Generate forward blob
	layer1.Forward(this->blob_bottom_vec, &(this->blob_top_vec));
	vector<Dtype> sequence;
	for(int i = 0, offset = 0; i < 2; i++)
	{
		for(int j = 0; j < top_dims; j++) sequence.push_back(this->blob_top->cpu_data()[offset++]);
	}
	
	// Calculate ground truth
	Caffe::set_random_seed(1701);
	vector<int> shuffle(4);
	for(int i = 0; i < 4; i++) shuffle[i] = i;
	
	std::random_shuffle(shuffle.begin(), shuffle.end(), SelectingLayerTest<TypeParam>::Random_Caffe);
	
	vector<Dtype> sequence_gt;
	for(int i = 0; i < 2; i++)
	{
		for(int j = 0; j < spatial_dims; j++)
		{
			Dtype value = this->blob_bottom->cpu_data()[i * bottom_dims + shuffle[0] * spatial_dims + j] +
				this->blob_bottom->cpu_data()[i * bottom_dims + shuffle[1] * spatial_dims + j] +
				this->blob_bottom->cpu_data()[i * bottom_dims + shuffle[2] * spatial_dims + j];
			sequence_gt.push_back(value);
		}
	}
	
	EXPECT_EQ(sequence.size(), sequence_gt.size());
	for(int i = 0; i < sequence.size(); i++)
	{
		EXPECT_EQ(sequence[i], sequence_gt[i]);
	}
}


TYPED_TEST(SelectingLayerTest, TestGradient)
{
	typedef typename TypeParam::Dtype Dtype;
	
	LayerParameter layer_param;
	SelectingParameter* selecting_param = layer_param.mutable_selecting_param();
	selecting_param->set_num_output(2);
	selecting_param->set_group_size(2);
	
	SelectingLayer<Dtype> layer(layer_param);
	
	GradientChecker<Dtype> checker(1e-2, 1e-2);
	checker.CheckGradientExhaustive(&layer, &(this->blob_bottom_vec), &(this->blob_top_vec));
}

TYPED_TEST(SelectingLayerTest, TestGradientOneGroup)
{
	typedef typename TypeParam::Dtype Dtype;
	
	LayerParameter layer_param;
	SelectingParameter* selecting_param = layer_param.mutable_selecting_param();
	selecting_param->set_num_output(1);
	selecting_param->set_group_size(3);
	
	SelectingLayer<Dtype> layer(layer_param);
	
	GradientChecker<Dtype> checker(1e-2, 1e-2);
	checker.CheckGradientExhaustive(&layer, &(this->blob_bottom_vec), &(this->blob_top_vec));
}

} // namespace caffe
