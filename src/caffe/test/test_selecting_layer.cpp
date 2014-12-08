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
	selecting_param->set_num_output(4);
	selecting_param->set_group_size(2);
	selecting_param->set_distinct(false);
	
	SelectingLayer<Dtype> layer(layer_param);
	layer.SetUp(this->blob_bottom_vec, &(this->blob_top_vec));
	EXPECT_EQ(this->blob_bottom->width(), this->blob_top->width());
	EXPECT_EQ(this->blob_bottom->height(), this->blob_top->height());
	EXPECT_EQ(this->blob_top->channels(), 4);
	EXPECT_EQ(this->blob_bottom->num(), this->blob_top->num());
}

TYPED_TEST(SelectingLayerTest, TestSetupDistinct)
{
	typedef typename TypeParam::Dtype Dtype;
	LayerParameter layer_param;
	
	SelectingParameter *selecting_param = layer_param.mutable_selecting_param();
	selecting_param->set_num_output(1);
	selecting_param->set_group_size(3);
	
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
	
	const int bottom_num = this->blob_bottom->num();
	const int num_output = 4;
	const int group_size = 2;
	const int bottom_channels = this->blob_bottom->channels();
	
	LayerParameter layer_param;
	SelectingParameter *selecting_param = layer_param.mutable_selecting_param();
	selecting_param->set_num_output(num_output);
	selecting_param->set_group_size(group_size);
	selecting_param->set_distinct(false);
	
	Caffe::set_random_seed(1701);
	SelectingLayer<Dtype> layer1(layer_param);
	layer1.SetUp(this->blob_bottom_vec, &(this->blob_top_vec));

	const int spatial_dims = this->blob_bottom->height() * this->blob_bottom->width();
	const int top_dims = this->blob_top->channels() * spatial_dims;
	const int bottom_dims = this->blob_bottom->channels() * spatial_dims;

	// Generate forward blob
	layer1.Forward(this->blob_bottom_vec, &(this->blob_top_vec));
	
	vector<Dtype> sequence;
	for(int i = 0, offset = 0; i < bottom_num; i++)
	{
		for(int j = 0; j < top_dims; j++) sequence.push_back(this->blob_top->cpu_data()[offset++]);
	}
	
	// Calculate ground truth
	Caffe::set_random_seed(1701);
	vector<vector<int> > shuffles;
	vector<int> shuffle(bottom_channels);
	for(int i = 0; i < bottom_channels; i++) shuffle[i] = i;
	for(int i = 0; i < bottom_channels; i++)
	{
		std::random_shuffle(shuffle.begin(), shuffle.end(),
			SelectingLayerTest<TypeParam>::Random_Caffe);
		shuffles.push_back(shuffle);
	}
	
	vector<Dtype> sequence_gt;
	for(int i = 0; i < bottom_num; i++)
	{
		for(int och = 0; och < num_output; och++)
		{
			vector<int> &shuffle = shuffles[och];
			
			for(int j = 0; j < spatial_dims; j++)
			{
				Dtype value = 0;
				
				for(int g = 0; g < group_size; g++)
				{
					value += this->blob_bottom->cpu_data()[i * bottom_dims + 
						shuffle[g] * spatial_dims + j];
				}
				
				sequence_gt.push_back(value);
			}
		}
	}
	
	EXPECT_EQ(sequence.size(), sequence_gt.size());
	for(int i = 0; i < sequence.size(); i++)
	{
		EXPECT_EQ(sequence[i], sequence_gt[i]);
	}
}

TYPED_TEST(SelectingLayerTest, TestForwardDistinct)
{
	typedef typename TypeParam::Dtype Dtype;
	
	const int bottom_num = this->blob_bottom->num();
	const int num_output = 1;
	const int group_size = 3;
	const int bottom_channels = this->blob_bottom->channels();
	
	LayerParameter layer_param;
	SelectingParameter *selecting_param = layer_param.mutable_selecting_param();
	selecting_param->set_num_output(num_output);
	selecting_param->set_group_size(group_size);
	
	Caffe::set_random_seed(1701);
	SelectingLayer<Dtype> layer1(layer_param);
	layer1.SetUp(this->blob_bottom_vec, &(this->blob_top_vec));

	int spatial_dims = this->blob_bottom->height() * this->blob_bottom->width();
	int top_dims = this->blob_top->channels() * spatial_dims;
	int bottom_dims = this->blob_bottom->channels() * spatial_dims;

	// Generate forward blob
	layer1.Forward(this->blob_bottom_vec, &(this->blob_top_vec));
	vector<Dtype> sequence;
	for(int i = 0, offset = 0; i < num_output; i++)
	{
		for(int j = 0; j < top_dims; j++) sequence.push_back(this->blob_top->cpu_data()[offset++]);
	}
	
	// Calculate ground truth
	Caffe::set_random_seed(1701);
	vector<int> shuffle(bottom_channels);
	for(int i = 0; i < bottom_channels; i++) shuffle[i] = i;
	
	std::random_shuffle(shuffle.begin(), shuffle.end(), SelectingLayerTest<TypeParam>::Random_Caffe);
	
	vector<Dtype> sequence_gt;
	for(int i = 0; i < num_output; i++)
	{
		for(int j = 0; j < spatial_dims; j++)
		{
			Dtype value = 0;
			
			for(int g = 0; g < group_size; g++)
			{
				value += this->blob_bottom->cpu_data()[i * bottom_dims + shuffle[g] * spatial_dims + j];
			}
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
	selecting_param->set_num_output(4);
	selecting_param->set_group_size(2);
	selecting_param->set_distinct(false);

	SelectingLayer<Dtype> layer(layer_param);
	
	GradientChecker<Dtype> checker(1e-2, 1e-2);
	checker.CheckGradientExhaustive(&layer, &(this->blob_bottom_vec), &(this->blob_top_vec));
}

TYPED_TEST(SelectingLayerTest, TestGradientDistinct)
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
