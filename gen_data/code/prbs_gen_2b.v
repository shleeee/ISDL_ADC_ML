//////////////////////////////////////////
//	PRBS Pattern Generator		//
//	Author : SCJANG			//
//	Update : 2012.06.28		//
//////////////////////////////////////////

module prbs_gen_2b(
	input		arstb,		// Async resetb
	input		rstb,		//  Sync resetb
	input		clk,		// Clock
	input		prbs_en,	// 1: PRBS out, 0: toggle out
	input		inv,		// 1: invert output polarity, 0: normal
	input	[1:0]	ptrn_sel,	// 00:PRBS7, 01:PRBS10, 10:PRBS15, 11:PRBS31
	
	output [1:0]	out		// PRBS out
	);

	reg	[30:0]	ptrn;

	always @(posedge clk or negedge arstb)
		begin
		if(!arstb)	ptrn <= 31'b010_1010_1010_1010_1010_1010_1010_1010;
		else if(!rstb)	ptrn <= 31'b010_1010_1010_1010_1010_1010_1010_1010;
		else
			begin
			if(prbs_en)
				begin
				case(ptrn_sel)
					2'b00 : ptrn <= {ptrn[30:7],(ptrn[0]^ptrn[1]),ptrn[6:1]};
					2'b01 : ptrn <= {ptrn[30:10],(ptrn[0]^ptrn[3]),ptrn[9:1]};
					2'b10 : ptrn <= {ptrn[30:15],(ptrn[0]^ptrn[1]),ptrn[14:1]};
					2'b11 : ptrn <= {(ptrn[0]^ptrn[3]),ptrn[30:1]};
					default : ptrn <= {(ptrn[0]^ptrn[3]),ptrn[30:1]};
				endcase
				end
			else	ptrn[0] <= ~ptrn[0];
			end
		end

	assign	out = inv ? ~ptrn[1:0] : ptrn[1:0];

endmodule
