/*
Bit Operations
*/
// Return data.bit value
#define get_bit(data, bit) ( (data>>bit) & 1)

// Set data.bit to 1
#define set_bit_high(data, bit) data |= (1 << bit)

// Set data.bit to 0
#define set_bit_low(data, bit)  data &= ~(1 << bit)

// Return bit range
#define get_bit_range(data, start_bit, end_bit) ((data >> start_bit) & ((1 << (end_bit - start_bit)) - 1))
