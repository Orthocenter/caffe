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
	distinct = selecting_param.distinct();
}

template <typename Dtype>
void SelectingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
	vector<Blob<Dtype>*>* top)
{
	num = bottom[0]->num();
	channels = bottom[0]->channels();
	width = bottom[0]->width();
	height = bottom[0]->height();

	this->blobs_.resize(1);
	this->blobs_[0].reset(new Blob<Dtype>(
		1, num_output, group_size, 1));

	if(distinct)
	{
		int total_output = group_size * num_output;
		
		CHECK(group_size * num_output <= channels) << "Group_size * num_output cannot be GREATER than input channels";
		
		vector<int> shuffle(channels);
		for(int i = 0 ; i < channels; i++) shuffle[i] = i;
		std::random_shuffle(shuffle.begin(), shuffle.end(), Random_Caffe);
		
		Dtype *data = this->blobs_[0]->mutable_cpu_data();
		for(int i = 0; i < total_output; i++) data[i] = round(shuffle[i]);
	}
	else
	{
		vector<int> shuffle(channels);
		for(int i = 0; i < channels; i++) shuffle[i] = i;
		
		for(int i = 0; i < num_output; i++)
		{
			std::random_shuffle(shuffle.begin(), shuffle.end(), Random_Caffe);
			
			Dtype *data = this->blobs_[0]->mutable_cpu_data() + this->blobs_[0]->offset(0, i);
			for(int j = 0; j < group_size; j++) data[j] = round(shuffle[j]);
		}
	}
	
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
		for(int i = 0; i < num_output; i++)
		{
			const Dtype *shuffle = this->blobs_[0]->cpu_data() + this->blobs_[0]->offset(0, i);
			
			// set output_data to 0
			Dtype *output_data = top_data + (*top)[0]->offset(n, i);
			caffe_set(spatial_dims,  Dtype(0), output_data);

			// add all channels in i_th group into output channel with scale (1 / group_size)
			for(int j = 0; j < group_size; j++)
			{
				caffe_axpy<Dtype>(spatial_dims, Dtype(1. / group_size), 
					bottom_data + bottom[0]->offset(n, round(shuffle[j])), output_data);
			}
		}
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
		
		caffe_set((*bottom)[0]->count(), Dtype(0), bottom_diff);
		
		for(int n = 0; n < num; n++)
		{
			for(int i = 0; i < num_output; i++)
			{
				const Dtype *shuffle = this->blobs_[0]->cpu_data() + this->blobs_[0]->offset(0, i);
				const Dtype* top_diff = top[0]->cpu_diff() + top[0]->offset(n, i);
				for(int j = 0; j < group_size; j++)
				{
					caffe_axpy<Dtype>(spatial_dims, Dtype(1. / group_size), top_diff, 
						bottom_diff + (*bottom)[0]->offset(n, round(shuffle[j])));
				}
			}
		}
	}
}

INSTANTIATE_CLASS(SelectingLayer);
} // namespace caffe
