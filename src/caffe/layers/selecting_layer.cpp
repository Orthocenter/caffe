#include <vector>
#include <algorithm>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void SelectingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
	vector<Blob<Dtype>*>* top)
{
	SelectingParameter selecting_param = this->layer_param_.selecting_param();
	
	num_output = selecting_param.num_output();
	group_size = selecting_param.group_size();
}

template <typename Dtype>
void SelectingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
	vector<Blob<Dtype>*>* top)
{
	count = bottom[0]->count();
	num = bottom[0]->num();
	channels = bottom[0]->channels();
	width = bottom[0]->width();
	height = bottom[0]->height();

	CHECK(group_size * num_output <= channels) << "Group_size * num_output cannot be GREATER than input channels";
	
	shuffle.resize(channels);
	for(int i = 0; i < channels; i++) shuffle[i] = i;
	std::random_shuffle(shuffle.begin(), shuffle.end());
	
	(*top)[0]->Reshape(num, num_output, height, width);
}

template <typename Dtype>
void SelectingLayer<Dtype>::Forward_cpu(
	const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top)
{
	Dtype* top_data = (*top)[0]->mutable_cpu_data();
	const Dtype* bottom_data = bottom[0]->cpu_data();
	
	int spatial_dims = width * height;
	for(int n = 0; n < num; n++)
	{
		for(int i = 0, shuffle_offset = 0; i < num_output; i++)
		{
			// copy the first channel in i_th group into output channel
			Dtype *output_data = top_data + (*top)[0]->offset(n, i);
			caffe_copy(spatial_dims, 
				bottom_data + bottom[0]->offset(n, shuffle[shuffle_offset++]), output_data);
		
			// add other channels in i_th group into output channel
			for(int j = 1; j < group_size; j++)
			{
				caffe_axpy<Dtype>(spatial_dims, 1., 
					bottom_data + bottom[0]->offset(n, shuffle[shuffle_offset++]), output_data);
			}
		}
	
		// average
		double scale_factor = 1. / group_size;
		caffe_axpy<Dtype>(count, scale_factor - 1, top_data + (*top)[0]->offset(n), 
			top_data + (*top)[0]->offset(n));
	}
}

template <typename Dtype>
void SelectingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
	const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom)
{
	if(propagate_down[0])
	{
		int spatial_dims = width * height;
		Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
		const Dtype* top_diff = top[0]->cpu_diff();
		
		for(int n = 0; n < num; n++)
		{
			for(int i = 0, shuffle_offset = 0; i < num_output; i++)
			{
				for(int j = 0; j < group_size; j++)
				{
					caffe_copy(spatial_dims, top_diff, 
						bottom_diff + (*bottom)[0]->offset(n, shuffle[shuffle_offset++]));
				}
				top_diff += top[0]->offset(0, 1);
			}
		}
	}
}

INSTANTIATE_CLASS(SelectingLayer);
} // namespace caffe
