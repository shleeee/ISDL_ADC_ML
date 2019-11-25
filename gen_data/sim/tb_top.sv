// Testbench 
// 2019.11.15
// Young-Ha Hwang
`timescale 		1ns/10fs
//`timescale 		1ns/10ps
//`define 		IDEAL_REF_
//`define 		POSIM_
//`define 		IDEAL_DCO_
//`define		PERR

module tb_top();

	reg clk, rstb;

	// NRZ
//	prbs_gen_1b i_prbs_gen(rstb, rstb, clk, 1'b1, 1'b0, 2'b11, prbs_out);
//	real ch_in, ch_out;
//	assign ch_in = prbs_out ? 0.5 : -0.5;

	//PAM4	
	wire  prbs_out;
//	prbs_gen_2b i_prbs_gen(rstb, rstb, clk, 1'b1, 1'b0, 2'b11, prbs_out);
	//prbs_gen_1b i_prbs_gen_msb(rstb, rstb, clk, 1'b1, 1'b0, 2'b10, prbs_out[1]);
	//prbs_gen_1b i_prbs_gen_lsb(rstb, rstb, clk, 1'b1, 1'b0, 2'b11, prbs_out[0]);
	prbs_gen_1b i_prbs_gen(rstb, rstb, clk, 1'b1, 1'b0, 2'b10, prbs_out);

	real ch_in, ch_out;
	//assign ch_in = (prbs_out==2'b00) ? -3 :
	//	       (prbs_out==2'b01) ? -1 :
	//	       (prbs_out==2'b10) ? 1 : 3;
	assign ch_in = (prbs_out==1'b0) ? -3 : 3;

	//channel loss : 10, 17, 23, 27 dB @ 5GHz	       
	channel_27dB i_ch(ch_in,ch_out);

	wire [5:0] adc_out;
	reg adc_clk;
	//reg dac_clk;
	adc_6b i_adc(1'b0, adc_clk, ch_out, adc_out);

//dac_6b i_dac(1'b0, dac_clk, adc_out, dac_out);

	initial clk <= 1'b1; 	
//	initial adc_clk<=1'b1;

	//data export
	integer f_in, f_out;	
	initial f_in=$fopen("./data_in.txt", "w"); 
	initial f_out=$fopen("./data_out.txt","w");


parameter CLK_PERIOD = 0.1;
parameter TIMESTEP = CLK_PERIOD/20;
//parameter DELAY = 0.02+6*TIMESTEP;
`include "./param.v"
`include "./reg.v"
parameter SIM_TIME=CLK_PERIOD*800000;
parameter jitter = 0.01;
//integer seed;

	always #(CLK_PERIOD/2) clk<=~clk;
	always @(posedge clk or negedge clk) #(TIMESTEP*DELAY) adc_clk<=clk;
//	always @(posedge adc_clk or negedge adc_clk) #(DELAY) dac_clk<=adc_clk;
//	assign #(DELAY) adc_clk=clk;

	// jittery clock
//	assign #(DELAY+jitter*$dist_normal(seed,0,1)) adc_clk = clk;
//	always #(DELAY+jitter*$dist_normal(seed,0,1)) adc_clk<=clk;

	always #CLK_PERIOD $fwrite(f_in, "%b\n", prbs_out); 
	always #CLK_PERIOD $fwrite(f_out, "%b\n", adc_out);






initial
begin

		rstb=1'b1;
		#0.01 
		rstb=1'b0;
		#0.01
		rstb=1'b1;


end

// Simulation start and finish
initial
begin
		$shm_open		("top.shm");
		$shm_probe		("AC");	 	// probe all
		//$shm_probe		(tb_top.i_adpll_dig_top.ssc_frac);	 	
		#(SIM_TIME)
		$finish(2);
		$shm_close();	
end


endmodule
