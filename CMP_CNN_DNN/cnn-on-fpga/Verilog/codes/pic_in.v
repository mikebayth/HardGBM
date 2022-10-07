`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date:    15:03:52 03/11/2019 
// Design Name: 
// Module Name:    pic_in 
// Project Name: 
// Target Devices: 
// Tool versions: 
// Description: 
//
// Dependencies: 
//
// Revision: 
// Revision 0.01 - File Created
// Additional Comments: 
//
//////////////////////////////////////////////////////////////////////////////////
module pic_in
#(
	parameter									bits						=	16		,	//quantization bit number
	parameter									bits_shift				=	4		,	//we can shift but not multy
	parameter									channel_num 			= 	1		,	//number of the channel of the picture
	parameter									channel_height_num	=	4		,//4�������ɵĴ�Ĵ��ڵĸ�
	parameter									channel_length_num	=	4		,//�ĳ�
	
	parameter									channel_all_num		=	16		,	//channel_paralell_num*channel_num
	parameter									bits_channel			=	256	,	//bit number of all data,
	//change																							//which is bits*channel_all_num
    parameter									length   				=  60	,	//length of input picture 
	parameter									length_2					=	6		,	//bit number of length
	//change
	parameter									height					=	60	,	//height of input picture
	parameter									height_2					=	6		,	//bit number of height

    parameter                                   data_size               = 3600,
    parameter                                   data_size_2             = 12, 
	
	//change
	parameter									filter_size 			= 	3		,	//size of filter is n*n,default 5*5
	parameter									filter_size_2			=	2		,	//the bits of the filter size
	parameter 									stride			= 	4		,	//stride of conv in the first conv_layer

	parameter									weight_num				=	9		,// filter_size^2
	parameter									weight_num_2			=	4		,
	parameter									conv_num					=	4		,//the number of conv kernel
	parameter									state_reset				=	0		,	//state to reset
	parameter									state_output			=	1			//state to output data
	)		
	(		
	input			 											clk_in,
	input														rst_n,	
	input														start,				//start to output
	output reg		[bits_channel-1:0]				map,					//picture data
	output reg												ready,					
	output reg		[(conv_num<<bits_shift)-1:0]	weight
    );

//***************************************************************************
//output control,output these blocks one by one,parallel output per channel**
//***************************************************************************
//**************************picture_parameter********************************
reg														state								;	//two states:reset,output
reg						[length_2-1:0]				cnt_length_start				;	//length of the start data
reg						[filter_size_2-1:0]		cnt_length_offset				;	//length offset of current data
reg						[height_2-1:0]				cnt_height_start				;	//height of the start data
reg						[filter_size_2-1:0]		cnt_height_offset				;	//height offset of current data
//**************************weight_parameter*********************************
reg						[weight_num_2-1:0]		cnt_weight_output				;
always@(posedge clk_in or negedge rst_n)
begin
	if(~rst_n)
	begin
		state										<=							state_reset			;
	end
	else
	begin
		if(state == state_reset)
		begin
			cnt_length_start					<=							0						;
			cnt_length_offset					<=							0						;
			cnt_height_start					<=							0						;
			cnt_height_offset					<=							0						;
			cnt_weight_output					<=							0						;
			ready									<=							1'b0					;
			if(start)
				state								<=							state_output		;
		end
		else if(state == state_output)
		begin
			ready										<=							1'b1											;
			//************************************************************************
			///**************************picture output*******************************
			//************************************************************************
			if(cnt_height_offset == filter_size-1)						
			begin
				cnt_height_offset					<=							0												;
				if(cnt_length_offset == filter_size-1)					
				begin
					cnt_length_offset				<=							0												;
					if(length - filter_size - cnt_length_start < stride)
					begin
						cnt_length_start		<=							0												;
						if(height - filter_size - cnt_height_start < stride)
						begin
							state						<=							state_reset									;	
						end
						else
						begin
							cnt_height_start		<=							cnt_height_start + stride		;
						end
					end
					else
					begin
						cnt_length_start			<=							cnt_length_start + stride		;
					end
				end
				else
				begin
					cnt_length_offset				<=							cnt_length_offset + 1'b1				;
				end
			end
			else
			begin
				cnt_height_offset					<=							cnt_height_offset + 	1'b1				;
			end
			
			if(cnt_weight_output == weight_num-1)
			begin
				cnt_weight_output 				<=							0									;
			end
			else
			begin
				cnt_weight_output					<=							cnt_weight_output + 1'b1	;
			end
		end
	end
end

wire 				[data_size_2-1:0]					addra			[0:channel_all_num-1]					;	
wire				[bits-1:0]				douta			[0:channel_all_num-1]						;
genvar i,j,k;
generate
	for(i = 0; i < channel_num; i = i + 1)
	begin:outdata_channel
	//坌一坷积核的坌一输出通靓下，应一次输�?16个数值以供应第一层坷积层计算，此处的j*channel_height_num+k坳指定了并行计算的index
		for(j = 0; j < channel_height_num; j = j + 1)
		begin:outdata_height
			for(k = 0; k < channel_length_num; k = k + 1)
			begin:outdata_length
			//输入的是该数杮的地址，输出的是该数杮的�??
				input_data Input_Data (
				  .clka(clk_in), // input clka
				  .addra(addra[i*channel_height_num*channel_length_num+j*channel_length_num+k]), 
				  .douta(douta[i*channel_height_num*channel_length_num+j*channel_length_num+k]) 
				);

				always@(posedge clk_in)
				begin
					map[((i*channel_height_num*channel_length_num+j*channel_length_num+k)<<bits_shift)+bits-1:((i*channel_height_num*channel_length_num+j*channel_length_num+k)<<bits_shift)] <= douta[i*channel_height_num*channel_length_num+j*channel_length_num+k]	;		
				end
				assign addra[i*channel_height_num*channel_length_num+j*channel_length_num+k]	= i*length*height+(cnt_height_start+cnt_height_offset+j)*length + cnt_length_start+cnt_length_offset + k	;
			end
		end
	end
endgenerate
wire					[(conv_num<<bits_shift)-1:0]		weight_temp	;
weight1 Weight1 (
  .clka(clk_in), 
  .addra(cnt_weight_output), 
  .douta(weight_temp) 
);

always@(posedge clk_in)
begin
	weight							<=							weight_temp				;
end
endmodule
