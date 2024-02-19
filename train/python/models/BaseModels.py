import torch

class FCModel(torch.nn.Module):
    def __init__(self, pmt_N, **kwargs):
        super().__init__()

        self.pmt_N = pmt_N

        self.fc1 = torch.nn.Sequential(
            torch.nn.Linear(pmt_N, 4096),
            torch.nn.ReLU(), torch.nn.Dropout(),
            torch.nn.Linear(4096, 1024),
            torch.nn.ReLU(), torch.nn.Dropout(),
            torch.nn.Linear(1024, 1024),
            torch.nn.ReLU(), torch.nn.Dropout(),
            torch.nn.Linear(1024, 1024),
            torch.nn.Sigmoid(),
            torch.nn.Linear(1024, 3),
        )

    def forward(self, pmt_q, pmt_t, pmt_pos, pmt_dir):
        batch, pmt_N = pmt_q.shape

        x = self.fc1(pmt_q)
        return x

class ClassicOpticalModel(torch.nn.Module):
    def __init__(self, pmt_N, **kwargs):
        super().__init__()

        self.pmt_N = pmt_N
        self.optModelOrder = 3

        self.wgtN = torch.nn.Parameter(torch.randn(1))
        self.wgtP = torch.nn.Parameter(torch.randn(1))
        self.wgtC = torch.nn.Parameter(torch.randn(pmt_N, self.optModelOrder))

    def forward(self, pmt_q, pmt_t, pmt_pos, pmt_dir, vtxPos):
        batch, pmt_N = pmt_q.shape
        ## pmtPos shape: (batch, pmt_N, 3)
        ## vtxPos shape: (batch, 3)

        ## Get the dimming factors - the optical model

        ## Get distance of the vertex from PMTs, shape = (batch, pmt_N)
        ## dxyz = vv - pmt_pos
        ##        vv      shape :  (batch, pmt_N, 3)
        ##        pmt_pos shape => (batch,     1, 3)
        ##        dxyz    shape :  (batch, pmt_N, 3) by broadcasting rule
        dxyz = vtxPos.view(-1,1,3) - pmt_pos

        ## First dimming factor by distance from the vertex
        dr2 = (dxyz**2).sum(dim=-1)
        dr = torch.sqrt(dr2) + 1e-9
        dimmDist = torch.sigmoid(self.wgtN)/dr2 * torch.exp(dr*self.wgtP.view(1,1))

        ## Second dimming factor by the angle to the PMTs normal vector
        ## Get the cos(theta) of vertex and PMT's normal vector
        dxyz /= dr.view(-1, self.pmt_N, 1) # broadcast the latest item
        ### cosT* shape = (batch, pmt_N)
        cosT1 = (dxyz*pmt_dir.view(1,-1,3)).sum(dim=[-1])
        cosTs = [torch.ones(cosT1.shape, device=cosT1.device), cosT1]
        for i in range(2, self.optModelOrder):
            cosTs.append(cosT1**(i))
        cosTs = torch.stack(cosTs, dim=-1)
        dimmAngl = (cosTs*self.wgtC.view(1, *self.wgtC.shape)).sum(dim=-1)

        ## Overall dimming factor, product of dimmDist and dimmAngl
        qFracPred = dimmAngl*dimmDist
        #qFracPred = dimmDist

        return qFracPred
