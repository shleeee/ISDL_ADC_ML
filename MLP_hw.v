//////////////////////////////////////////
//	Author : SHROH			//
//	Update : 2019.12.02		//
//////////////////////////////////////////

module MLP_hw
	#(parameter INPUT_BIT = 6, INPUT_NUM = 5, NEURON_NUM = 8, OUTPUT_NUM = 4
	 , WEIGHT_BIT = 6, BIAS_BIT = 6)
	(
	input	[INPUT_BIT-1:0]	data_in [0:INPUT_NUM-1],
	input	write_en,
	input	cal_en
	input	addr_w,
	input	addr_b,
	input	[WEIGHT_BIT-1:0] weight,
	input	[BIAS_BIT-1:0]	 bias,
	input	arstb,
	
	output	data_out [0:OUTPUT_NUM-1];
	);

	reg	[WEIGHT_BIT-1:0]	weight1 [0:INPUT_NUM][0:NEURON_NUM-1];
	reg	[BIAS_BIT-1:0]		bias1 [0:NEURON_NUM-1];
	reg	[WEIGHT_BIT-1:0]	weight2 [0:NEURON_NUM-1][0:OUTPUT_NUM];
	reg	[BIAS_BIT-1:0]		bias2 [0:OUTPUT_NUM-1];
	reg	[]			neuron [0:NEURON_NUM-1];
	reg				

always @(posedge clk or negedge arstb) begin
	if(!arstb) begin
		for(integer i=0; i < NEURON_NUM; i=i+1) begin
			for(integer j=0; j < INPUT_NUM; j=j+1) begin
				weight1[j][i] = 0;
			end
			bias1[i] = 0;
		end
		for(integer i=0; i < OUTPUT_NUM; i=i+1) begin
			for(integer j=0; j < NEURON_NUM; j=j+1) begin
				weight2[j][i] = 0;
			end
			bias2[i] = 0;
		end
	end //end reset
	else begin
		if(write_en) begin
			weight1[addr_w] <= weight;
			bias1[addr_b] <= bias;
		end
		else begin
			if(cal_en) begin
				for(integer j=0; j < NEURON_NUM; j=j+1) begin
					for(integer i=0; i < INPUT_NUM; i=i+1) begin
						neuron[j] <= neuron[j] + data_in[i]*weight1[i][j]; // calculation
					end
					neuron[j] <= neuron[j] + bias1[j];
				end
				for(integer j=0; j < OUTPUT_NUM; j=j+1) begin
					for(integer i=0; i < NEURON_NUM; i=i+1) begin
						outreg[j] <= outreg[j] + neuron[i]*weight2[i][j];
					end
					outreg[j] <= outreg[j] + bias2[j];
				end
			end
			else begin
			end
		end
	end
end
	
endmodule
