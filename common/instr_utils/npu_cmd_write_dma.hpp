#ifndef __NPU_CMD_WRITE_DMA_HPP__
#define __NPU_CMD_WRITE_DMA_HPP__

#include "npu_cmd.hpp"

struct npu_dma_block_cmd : public npu_cmd{
    uint32_t col;
    uint32_t row;
    uint32_t bd_id;
    uint32_t op_size;
    uint32_t buffer_length;
    uint32_t buffer_offset;
    uint32_t packet_enable;
    uint32_t out_of_order_id;
    uint32_t packet_id;
    uint32_t packet_type;
    uint32_t repeat_cnt;
    uint32_t issue_token;

    bool is_linear;
    uint32_t dim0_size;
    uint32_t dim0_stride;
    uint32_t dim1_size;
    uint32_t dim1_stride;
    uint32_t burst_size;
    uint32_t dim2_size;
    uint32_t dim2_stride;

    uint32_t iter_size;
    uint32_t iter_stride;
    uint32_t next_bd_id;
    uint32_t use_next_bd;
    uint32_t valid_bd;

    uint32_t get_lock_rel_val;
    uint32_t get_lock_rel_id;
    uint32_t get_lock_acq_enable;
    uint32_t get_lock_acq_val;
    uint32_t get_lock_acq_id;

    void dump_cmd(uint32_t *bd){
        assert(*bd == dma_block_write);
        LOG_VERBOSE(1, "bd_addr: " << bd);
        this->col = ((bd[1] >> bd_col_shift) & bd_col_mask);
        this->row = ((bd[1] >> bd_row_shift) & bd_row_mask);
        this->bd_id = ((bd[1] >> bd_id_shift) & bd_id_mask);
        // all words can be found in mlir-aie/lib/Dialect/AIEX/Transforms/AIEDmaToNpu.cpp close to line 577
        // word 0: unknown yet
        this->op_size = bd[2] >> 2; // 4 bytes per instruction
        // word 1: Buffer length
        this->buffer_length = bd[3];
        // word 2: Buffer offset
        this->buffer_offset = bd[4];
        // word 3: Packet information
        this->packet_enable = (bd[5] >> en_packet_shift) & en_packet_mask;
        this->out_of_order_id = (bd[5] >> out_of_order_shift) & out_of_order_mask;
        this->packet_id = (bd[5] >> packet_id_shift) & packet_id_mask;
        this->packet_type = (bd[5] >> packet_type_shift) & packet_type_mask;
        // word 4: D0
        this->is_linear = (bd[6] == 0);
        this->dim0_size = (bd[6] >> dim_size_shift) & dim_size_mask;
        this->dim0_stride = (bd[6] >> dim_stride_shift) & dim_stride_mask;
        this->dim0_stride += 1; // The saved value is the stride - 1

        // word 5: D1
        this->burst_size = 0x80000000 >> 30; // this is a constant value
        this->dim1_size = (bd[7] >> dim_size_shift) & dim_size_mask;
        this->dim1_stride = (bd[7] >> dim_stride_shift) & dim_stride_mask;
        this->dim1_stride += 1; // The saved value is the stride - 1
        // word 6: D2
        if (!this->is_linear){
            this->dim2_size = buffer_length / (this->dim0_size * this->dim1_size);
        }
        else{
            this->dim2_size = 0;
        }
        this->dim2_stride = (bd[8] >> dim_stride_shift) & dim_stride_mask;
        this->dim2_stride += 1; // The saved value is the stride - 1
        
        // word 7: D3, Iteration dimension
        this->iter_size = (bd[9] >> iter_size_shift) & iter_size_mask;
        this->iter_stride = (bd[9] >> iter_stride_shift) & iter_stride_mask;
        this->iter_stride += 1; // The saved value is the stride - 1
        this->iter_size += 1; // The saved value is the size - 1

        // word 8: Next BD, Lock information
        this->next_bd_id = (bd[10] >> next_bd_id_shift) & next_bd_id_mask;
        this->valid_bd = (bd[10] >> valid_bd_shift) & valid_bd_mask;

        // These informantion are provided but not used on NPU2
        this->get_lock_rel_val = (bd[10] >> get_lock_rel_val_shift) & get_lock_rel_val_mask;
        this->get_lock_rel_id = (bd[10] >> get_lock_rel_id_shift) & get_lock_rel_id_mask;
        this->get_lock_acq_enable = (bd[10] >> get_lock_acq_enable_shift) & get_lock_acq_enable_mask;
        this->get_lock_acq_val = (bd[10] >> get_lock_acq_val_shift) & get_lock_acq_val_mask;
        this->get_lock_acq_id = (bd[10] >> get_lock_acq_id_shift) & get_lock_acq_id_mask;
    }

    int print_cmd(uint32_t *bd, int line_number, int op_count){
        MSG_BONDLINE(INSTR_PRINT_WIDTH);
        // This is a dma bd
        instr_print(line_number++, bd[0], "DMA block write, OP count: " + std::to_string(op_count));
        // word 0:
        instr_print(line_number++, bd[1], "--Location: (row: " + std::to_string(row) + ", col: " + std::to_string(col) + ")");
        instr_print(-1, bd[1], "--BD ID: " + std::to_string(bd_id));

        // all words can be found in mlir-aie/lib/Dialect/AIEX/Transforms/AIEDmaToNpu.cpp close to line 577
        // word 0: unknown yet
        instr_print(line_number++, bd[2], "Operation size: " + std::to_string(this->op_size));
        // word 1: Buffer length
        instr_print(line_number++, bd[3], "--Buffer length: " + size_t_to_string(this->buffer_length));

        // word 2: Buffer offset
        instr_print(line_number++, bd[4], "--Buffer offset: " + size_t_to_string(this->buffer_offset));

        // word 3: Packet information
        if (this->packet_enable){
            // This is a packet
            instr_print(line_number++, bd[5], "--Packet enabled");
            instr_print(-1, bd[5], "--Out of order id: " + std::to_string(this->out_of_order_id));
            instr_print(-1, bd[5], "--Packet id: " + std::to_string(this->packet_id));
            instr_print(-1, bd[5], "--Packet type: " + std::to_string(this->packet_type));
        }
        else{
            instr_print(line_number++, bd[5], "Packet disabled");
        }

        // word 4: D0
        if (this->is_linear){
            instr_print(line_number++, bd[6], "A linear transfer, no D0");
        }
        else{
            instr_print(line_number++, bd[6], "--D0 size, stride: " + size_t_to_string(this->dim0_size) + ", " + size_t_to_string(this->dim0_stride));
        }

        // word 5: D1
        if (this->dim1_size == 0){
            instr_print(line_number++, bd[7], "--No D1");
        }
        else{
            instr_print(line_number++, bd[7], "--D1 size, stride: " + size_t_to_string(this->dim1_size) + ", " + size_t_to_string(this->dim1_stride));
        }

        // word 6: D2
        if (this->dim2_size == 0){
            instr_print(line_number++, bd[8], "--No D2");
        }
        else{
            instr_print(line_number++, bd[8], "--D2 stride: " + size_t_to_string(this->dim2_stride));
            instr_print(-1, bd[8], "--Inferred D2 size: " + size_t_to_string(this->dim2_size));
        }
        
        // word 7: D3, Iteration dimension
        if (this->iter_size == 0){
            instr_print(line_number++, bd[9], "--No Iteration dimension");
        }
        else{
            instr_print(line_number++, bd[9], "--Iteration size: " + size_t_to_string(this->iter_size));
            instr_print(-1, bd[9], "--Iteration stride: " + size_t_to_string(this->iter_stride));
        }
        
        // word 8: Next BD, Lock information
        instr_print(line_number++, bd[10], "--Next BD ID: " + std::to_string(this->next_bd_id));
        instr_print(-1, bd[10], "--Valid BD: " + std::to_string(this->valid_bd));
        instr_print(-1, bd[10], "--Lock relative value: " + size_t_to_string(this->get_lock_rel_val));
        instr_print(-1, bd[10], "--Lock relative id: " + std::to_string(this->get_lock_rel_id));
        instr_print(-1, bd[10], "--Lock acquire enable: " + std::to_string(this->get_lock_acq_enable));
        instr_print(-1, bd[10], "--Lock acquire value: " + size_t_to_string(this->get_lock_acq_val));
        instr_print(-1, bd[10], "--Lock acquire id: " + std::to_string(this->get_lock_acq_id));
        return line_number;
    }
    
    void to_npu(std::vector<uint32_t>& npu_seq){
        npu_seq.push_back(dma_block_write);
        npu_seq.push_back((row << bd_row_shift) | (col << bd_col_shift) | (bd_id << bd_id_shift) | 0x1D000);
        npu_seq.push_back(this->op_size * 4);
        npu_seq.push_back(this->buffer_length);
        npu_seq.push_back(this->buffer_offset);
        npu_seq.push_back((this->packet_enable << en_packet_shift) | (this->out_of_order_id << out_of_order_shift) | (this->packet_id << packet_id_shift) | (this->packet_type << packet_type_shift));
        npu_seq.push_back((this->dim0_size << dim_size_shift) | ((this->dim0_stride - 1) << dim_stride_shift));
        npu_seq.push_back(0x80000000 | (this->dim1_size << dim_size_shift) | ((this->dim1_stride - 1) << dim_stride_shift));
        npu_seq.push_back((0 << dim_size_shift) | ((this->dim2_stride - 1) << dim_stride_shift)); // dim 2 size is not required to be set
        npu_seq.push_back(((this->iter_size - 1) << iter_size_shift) | ((this->iter_stride - 1) << iter_stride_shift));
        npu_seq.push_back(
            (this->next_bd_id << next_bd_id_shift) | 
            (this->valid_bd << valid_bd_shift) | 
            (this->get_lock_rel_val << get_lock_rel_val_shift) | 
            (this->get_lock_rel_id << get_lock_rel_id_shift) | 
            (this->get_lock_acq_enable << get_lock_acq_enable_shift) | 
            (this->get_lock_acq_val << get_lock_acq_val_shift) | 
            (this->get_lock_acq_id << get_lock_acq_id_shift)
        );
    }

    int get_op_lines(){
        return 11;
    }
};

#endif
