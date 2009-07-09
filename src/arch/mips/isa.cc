/*
 * Copyright (c) 2009 The Regents of The University of Michigan
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met: redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer;
 * redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution;
 * neither the name of the copyright holders nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * Authors: Gabe Black
 */

#include "arch/mips/isa.hh"
#include "arch/mips/regfile/misc_regfile.hh"
#include "cpu/thread_context.hh"

namespace MipsISA
{

void
ISA::clear()
{
    miscRegFile.clear();
}

MiscReg
ISA::readMiscRegNoEffect(int miscReg)
{
    return miscRegFile.readRegNoEffect(miscReg);
}

MiscReg
ISA::readMiscReg(int miscReg, ThreadContext *tc)
{
    return miscRegFile.readReg(miscReg, tc);
}

void
ISA::setMiscRegNoEffect(int miscReg, const MiscReg val)
{
    miscRegFile.setRegNoEffect(miscReg, val);
}

void
ISA::setMiscReg(int miscReg, const MiscReg val, ThreadContext *tc)
{
    miscRegFile.setReg(miscReg, val, tc);
}

void
ISA::serialize(std::ostream &os)
{
    //miscRegFile.serialize(os);
}

void
ISA::unserialize(Checkpoint *cp, const std::string &section)
{
    //miscRegFile.unserialize(cp, section);
}

}
